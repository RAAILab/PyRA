# 파이썬으로 구현하는 로보어드바이저
=================================

## 빠르게 시작하기

### 이 프로젝트를 로컬 컴퓨터에 설치하고 싶은가요?

[아나콘다](https://www.anaconda.com/products/distribution)(또는 [미니콘다](https://docs.conda.io/en/latest/miniconda.html)), [깃](https://git-scm.com/downloads)을 설치하고, 텐서플로와 호환되는 GPU가 있다면 [GPU 드라이버](https://www.nvidia.com/Download/index.aspx)와 적절한 버전의 CUDA 및 cuDNN을 설치하세요(자세한 내용은 텐서플로우 설명서를 참조하세요).

그다음 터미널을 열고 다음 명령을 입력하여 이 저장소를 클론합니다(터미널 명령임을 표시하는 맨 앞의 `$` 기호는 입력하지 마세요):

    $ git clone https://github.com/RAAILab/PyRA.git
    $ cd pyRA

그다음 다음 명령을 실행합니다:

    $ conda create -n pyra python==3.9.7
    $ conda activate pyra
    $ pip install -r requirements.txt

본인이 설치한 CUDA 및 cuDNN 버전에 맞춰 적절한 버전의 pytorch를 설치합니다. 예를 들어, 1.13.1 버전의 pytorch와 11.7 버전의 CUDA를 설치하기 위해선 다음과 같은 명령을 실행합니다:

    $ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    
설치 명령어는 https://pytorch.org/get-started/previous-versions/ 를 참조하세요.

마지막으로 커널을 추가하고 주피터를 실행합니다:

    $ pip install ipykernel
    $ python -m ipykernel install --user --name=pyra
    $ jupyter notebook


# 자주 묻는 질문

**어떤 파이썬 버전을 사용해야 하나요?**

Python 3.9.7을 권장합니다. 위의 설치 가이드를 따르면 이 버전이 설치됩니다.

**어떤 pytorch 버전을 사용해야 하나요?**

Pytorch 1.13.1을 권장합니다. 위의 설치 가이드를 따르면 이 버전이 설치됩니다.
