import sys, subprocess

if __name__ == '__main__':
    packages = [\
    'matplotlib',\
    'pandas',\
    'numpy',\
    'scipy',\
    'tqdm',\
    'torch',\
    'torchvision']
    for mod in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', '--user', 'install', mod])