"""Modal images shared by the training, serving and ablation scripts.

`hopper_image`: CUDA 13 + torch 2.14 + transformers 5.17 + flash-linear-attention, with the upstream causal-conv1d CUDA
extension built for H100 (sm_90). `blackwell_image`: the same, rebuilt for B200 (sm_100). The extension's setup.py only
lists older architectures, so the build patches the gencode flag before installing.
"""
import modal
def _build(arch):
 import urllib.request,tarfile,pathlib,subprocess,os
 root=pathlib.Path('/tmp/causal-build');root.mkdir(exist_ok=True)
 archive=root/'source.tar.gz';urllib.request.urlretrieve('https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.7.0.tar.gz',archive)
 with tarfile.open(archive) as f:f.extractall(root,filter='data')
 source=root/'causal-conv1d-1.7.0';p=source/'setup.py';s=p.read_text();a=s.index('        cc_flag.append("-gencode")',s.index('# Check, if CUDA11'));b=s.index('    # HACK:',a)
 s=s[:a]+'        cc_flag += ["-gencode", "arch=compute_%s,code=sm_%s"]\n\n'%(arch,arch)+s[b:];p.write_text(s)
 env=dict(os.environ,CAUSAL_CONV1D_FORCE_BUILD='TRUE',MAX_JOBS='4',CC='gcc',CXX='g++',CUDAHOSTCXX='g++')
 subprocess.run(['pip','install','--no-build-isolation','--no-deps','--force-reinstall',str(source)],check=True,env=env)
def build_sm90():_build('90')     # Modal needs module-level functions for run_function
def build_sm100():_build('100')
_base=modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12').env({'HF_HOME':'/cache/huggingface'}).apt_install('build-essential').pip_install('torch==2.14.0','transformers==5.17.0','accelerate','pillow','sentencepiece','flash-linear-attention','torchvision','kernels==0.16.0','fastapi','uvicorn','ninja','packaging','setuptools','wheel')
hopper_image=_base.run_function(build_sm90)
blackwell_image=hopper_image.run_function(build_sm100)
