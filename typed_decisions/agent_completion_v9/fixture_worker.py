"""Linux-only bounded worker. Default-deny seccomp before dataset execution.

Parent starts this worker under UID/GID 65534, with only stdin/out/err inherited.
No filesystem opens, network, process creation, or executable loading is allowed
after initialization. AST/import restrictions are additional, not the sandbox.
"""
import ast
import builtins
import ctypes
import importlib
import json
import resource
import signal
import os
import socket
import sys

ALLOWED={'math','re','collections','itertools','functools','heapq','bisect','operator',
         'string','statistics','fractions','decimal','array','datetime'}
for module in ALLOWED:importlib.import_module(module)


def sandbox():
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(8,8))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    lib=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_load.argtypes=[ctypes.c_void_p]
    ctx=lib.seccomp_init(0x00050001)  # default EPERM
    if not ctx:raise RuntimeError('seccomp_init failed')
    names=['read','write','close','fstat','newfstatat','lseek','mmap','mprotect','munmap','mremap','brk',
           'rt_sigaction','rt_sigprocmask','rt_sigreturn','sigaltstack','setitimer','getitimer',
           'clock_gettime','gettimeofday','getpid','gettid','getrandom','futex','sched_yield',
           'exit','exit_group','restart_syscall']
    for name in names:
        number=lib.seccomp_syscall_resolve_name(name.encode())
        if number>=0 and lib.seccomp_rule_add(ctx,0x7fff0000,number,0)!=0:
            raise RuntimeError('seccomp rule failed')
    if lib.seccomp_load(ctx)!=0:raise RuntimeError('seccomp_load failed')


def vetted(source):
    tree=ast.parse(source)
    forbidden={'eval','exec','compile','open','input','breakpoint','getattr','setattr','delattr',
               'globals','locals','vars','dir','help','print','exit','quit'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and (node.id.startswith('__') or node.id in forbidden):
            raise ValueError('forbidden name')
        if isinstance(node,ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('private attribute')
        if isinstance(node,ast.Import) and any(a.name not in ALLOWED for a in node.names):
            raise ValueError('import not allowed')
        if isinstance(node,ast.ImportFrom) and (node.level or node.module not in ALLOWED or any(a.name.startswith('_') for a in node.names)):
            raise ValueError('import not allowed')
    return compile(tree,'<fixture>','exec')


def timeout(signum,frame):raise TimeoutError('fixture timeout')


def main():
    payload=json.load(sys.stdin)
    sandbox()
    if payload.get('sandbox_probe'):
        results={}
        for name,call in [('filesystem',lambda:open('/etc/passwd')),('network',lambda:socket.socket()),('process',lambda:os.fork())]:
            try:call();results[name]=False
            except PermissionError:results[name]=True
        print(json.dumps(results));return
    signal.signal(signal.SIGALRM,timeout)
    allowed_builtins={name:getattr(builtins,name) for name in
        ['abs','all','any','bool','bytes','chr','dict','divmod','enumerate','filter','float','format',
         'frozenset','hash','hex','int','isinstance','issubclass','iter','len','list','map','max','min','next',
         'oct','ord','pow','range','repr','reversed','round','set','slice','sorted','str','sum','tuple','zip',
         'Exception','ValueError','TypeError','IndexError','KeyError','AssertionError','StopIteration']}
    def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level or name not in ALLOWED:raise ValueError('import not allowed')
        return builtins.__import__(name,globals,locals,fromlist,level)
    allowed_builtins['__import__']=safe_import
    results=[]
    for source in payload['sources']:
        outcomes=[]
        for test in payload['tests']:
            try:
                code=vetted(payload.get('setup','')+'\n'+source)
                assertion=vetted(test)
                namespace={'__builtins__':allowed_builtins.copy()}
                signal.setitimer(signal.ITIMER_REAL,.15)
                exec(code,namespace,namespace);exec(assertion,namespace,namespace)
                outcomes.append('pass')
            except Exception as exc:outcomes.append(type(exc).__name__)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
        results.append(outcomes)
    print(json.dumps(results))


if __name__=='__main__':main()
