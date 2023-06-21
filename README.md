<div align=center>
    <h1>Segmentation Model Serving Basic</h1>
    <strong>Streamlit과 FastAPI로 만든 간단한 Hanbone Segmentation 페이지</strong>
    <br>
    <br>
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=FastAPI&logoColor=white">
    <img src="https://img.shields.io/badge/streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
    <img src="https://img.shields.io/badge/poetry-60A5FA?style=flat-square&logo=poetry&logoColor=white">
</div>

## 사용 방법

**Frontend**

    cd frontend
    streamlit run ui.py

**Backend**
    
    cd backend
    uvicorn server:app --reload


## 구조도
    .
    ├── README.md
    ├── backend
    │   ├── segmentation_module.py
    │   └── server.py
    ├── frontend
    │   └── ui.py
    ├── models
    │   ├── resnet-fpn.pt
    │   └── swin-fpn.pt
    ├── poetry.lock
    └── pyproject.toml

## 사용 가능 모델
**Encoder**

* Swin Transformer - B [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)]

* ResNet 101 [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)]

**Decoder**

* Feature_Pyramid_Networks(FPN) [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)]