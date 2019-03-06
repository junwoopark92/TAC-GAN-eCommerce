# Just Model It

![2019-02-28 1 42 47](https://user-images.githubusercontent.com/26558158/53541834-fe3ebd80-3b5e-11e9-851a-5fa6792df04a.png)

Just Model It 행사는 one day one model을 꿈꾸는 머신러닝 엔지니어에게 혹은 컴퓨터의 하드웨어 성능때문에 그동안 생각으로만 모델링하고 있던 사람에게 좋은기회입니다. 저 또한 회사에서 개인적인 모델을 학습할 수 없었기 때문에 이번 대회는 평소에 풀고 싶던 문제에 대해 모델링을 할수 있는 기회였습니다.
 
eCommerce에서 상품 메타정보 정제는 중요한 이슈입... 

<details><summary> 내용 더보기 ...</summary>
<p>
  
니다. 판매자와 소비자를 연결하는 쇼핑몰에서 많고 좋은 상품들은 소비자들에게 선택의 폭을 넓여주기 때문에 매력적인 요소 중 하나입니다. 하지만 하루에 수십개의 상품을 올리는 판매자에게 (1)**이미지 누락**이나 브랜드정보 누락 등 정제된 메타정보를 제공하는 상품등록을 기대하기는 힘들고 또한 판매자가 자신의 상품을 돋보이게 하기위해 (2)**상품 이미지에 홍보문구를 추가**하는 경우가 많은점은 상품본연의 정보와 일치하지 않는 상품 메타 데이터를 만들어냅니다. eCommerce에서는 판매자 또한 고객이기 때문에 판매자에게 위와같은 어뷰징을 제제하기 쉽지않고 실제로 하루에 등록되는 수천개의 상품에 대해 수작업으로는 정제할 수가 없습니다. 

(1),(2)의 문제를 해결하고 싶었고 상품의 메타 정보를 바탕으로 이미지를 생성하는 GAN 모델을 만들었습니다. (1)에 대해 Generator를 통해 상품명, 카테고리명, 브랜드, 그리고 속성등을 이용하여 상품이미지를 생성하고 이미지에 대해 real/fake를 구분하도록 학습된 Discriminator를 이용하여 (2)에 대해 상품과 관계 없는 어뷰징 이미지로 판별하도록 하였습니다.

관련하여 Text to Image synthesis대해 생성모델로 stackgan, stackgan++, tac-gan 논문을 찾았습니다. eCommerce 데이터를 이용한것은 없었기에 amazon product metadata를 학습시키기로 하였고 backend ai에 업로드하여 tac-gan을 프로토타입하였습니다. dataset에 존재하는 상품의수는 약 900만개이고 관련 리프 카테고리는 16000개 입니다. 이미지를 모두 다운받기위해 걸린시간은 약 200개의 쓰레드를 이용하여 1일이 걸렸고 이미지의 전체 용량은 260gb를 차지하였습니다. 개인컴퓨터로는 처리할수 없는 크기이기에 이번 대회에서 제공해주는 DGX를 충분히 이용하기 위해 전체적으로 큰 데이터셋을 선택하였습니다.

상품명을 이용하여 상품이미지를 생성하기 위해선 우선 상품명을 임베딩하여야 합니다. 위의 소개한 논문들은 skipthoughts라는 pretrained model을 사용하여 문장을 임베딩합니다. 하지만 eCommerce에서의 단어나 문장의 의미가 일반적인 의미와 다를수 있기 (예를들어 삼성 모니터 S23C550H 경우 모델명은 회화에선 의미없는 단어로 판단되지만 eCommerce에선 상품의 특징을 나타내는 단어로 작용) 때문에 임베딩모델을 직접 학습하였습니다. 먼저 out of vocabulary (OOV) 문제에 영향을 덜받는 Sentencepiece를 이용하여 문장의 각단어와 문자들을 인덱싱하였고 인덱싱된 문자들을 gensim의 doc2vec구현을 이용하여 문장 임베딩 모델을 학습하였습니다.

```
▁yamaha▁classical▁nylon▁string▁guitars▁yamaha▁c40▁full▁size▁nylon▁string▁classical▁guitar

[TOP-N SIM]
0	▁yamaha▁classical▁nylon▁string▁guitars▁yamaha▁c40▁full▁size▁nylon▁string▁classical▁guitar	0.89
1	▁yamaha▁yamaha▁cg192s▁spruce▁top▁classical▁guitar▁classical▁nylon▁string▁guitars	0.84
2	▁yamaha▁yamaha▁cg142c▁cedar▁top▁classical▁guitar▁classical▁nylon▁string▁guitars	0.82
3	▁classical▁nylon▁string▁guitars▁yamaha▁yamaha▁cg162c▁cedar▁top▁classical▁guitar	0.79
4	▁yamaha▁c40▁nylon▁string▁acoustic▁guitar▁bundle▁with▁hardshell▁case▁tuner▁instructional▁dvd▁strings▁pick▁card▁and▁polishing▁cloth▁classical▁nylon▁string▁guitars	0.79
5	▁classical▁nylon▁string▁guitars▁yamaha▁yamaha▁cg122ms▁spruce▁top▁classical▁guitar▁matte▁finish	0.76
6	▁yamaha▁classical▁nylon▁string▁guitars▁yamaha▁ncx900fm▁acoustic▁electric▁classical▁guitar▁flamed▁maple▁top	0.76
7	▁yamaha▁ntx700▁acoustic▁electric▁classical▁guitar▁yamaha▁classical▁nylon▁string▁guitars	0.75
8	▁classical▁nylon▁string▁guitars▁yamaha▁c40c▁full▁size▁classical▁acoustic▁guitar	0.74
9	▁yamaha▁classical▁nylon▁string▁guitars▁yamaha▁c40ii▁classical▁guitar	0.74

```

1차 시도로는 전체 900만개 데이터셋에 대하여 dgx의 모든 gpu를 이용하여 학습을 시도하였지만 1epoch를 처리하기 위해 약 1시간 가량이 소요되었고 또한 상품에 따른 이미지 차이가 커서 실제 데이터의 분포를 근사하지 못하는 모습을 보였습니다. 한달이라는 짧은시간에 전체를 하기엔 시간이 부족하다고 판단되서 guitar, nike, adidas, boot, helmet, cap, hat, dress 등 20여개의  카테고리에 대해 학습과 실험을 반복하였습니다.

같은 이미지에 다양한 캡션이 존재할수 있기 때문에 카테고리, 브랜드, 속성, 상품명등을 셔플링하는 방법으로 data augmentation을 하였습니다. 이렇게 진행하였을때 어느정도 상품의 형태와 상품명의 의미를 반영하였지만 전체적인 이미지 퀄리티가 나오지 않아 lsgan의 loss를 차용하여 학습하였고 전보다는 나은 이미지 퀄리티를 확인하였습니다. 
![good_loss_graph](https://user-images.githubusercontent.com/26558158/53537035-ec075400-3b4b-11e9-8c0d-7b1030b2037d.png)
![bad_loss_graph](https://user-images.githubusercontent.com/26558158/53537244-a72fed00-3b4c-11e9-8c02-94421012d1bf.png)

경험적으로 학습을 진행하면서 두번째 loss 그래프처럼 G의 loss와 D의 loss가 처음에는 함께 감소하다가 후에는 G loss는 발산하고 D로스는 수렴하는 경향을 보이면 더이상 이미지의 퀄리티가 좋아지지 않았습니다. 학습이 잘되는 경우에는 첫번째 G loss와 D loss가 증감하면서 감소하는 할때 학습이 잘되는 경향을 보입니다.

여러 카테고리의 결과중에서 guitar 카테고리가 문장에 따라 형태와 색생을 가장 잘 반영하여 자세한 실험결과를 추가했습니다. 이는 다른 카테고리에 비해 상품명이 상품이미지의 특성을 잘반영하고 있고 전체적으로 큰 형태변환이 덜한 카테고리이기 때문에 그렇습니다. 반대로 예를 들어 adidas 카테고리의 경우 신발, 티셔츠, 그리고 사람의 착용모습등 바꿔야할 서로간의 차이가 너무 크기때문에 학습이 잘안되는 경향을 보였습니다.

<img width="1000" alt="2019-02-28 1 46 44" src="https://user-images.githubusercontent.com/26558158/53541948-7c02c900-3b5f-11e9-8e93-5fd13648d739.png">

<img width="1000" alt="2019-02-28 1 49 17" src="https://user-images.githubusercontent.com/26558158/53541995-b40a0c00-3b5f-11e9-998d-4198748e9eae.png">

한달이라는 시간동안 처음으로 GAN학습을 하면서 빠르게 결과를 볼수 있었던것은 성능이 우수한 DGX시스템과 Backend AI의 환경덕분이었습니다. 개인이 사용하는 GTX GPU에 비하여 일반적으로 3~4배의 성능을 보이고 CUDA설치등에 시간을 소요하지않았기 때문에 시간을 절약할수 있었습니다. 추가적으로 조금 더 많은 카테고리에 대해 통합된 모델을 만들지 못한점이 약간의 아쉬움으로 남습니다. 행사가 끝난이후에도 추가적으로 진행해보고 싶습니다.
Backend AI의 내장된 Jupyter notebook과 개발환경이 데이터 디렉토리와 분리된점이 개인적으로 편하였습니다. 행사이름 처럼 Just model it한 좋은 경험이었습니다. 대회를 기획 및 주관해주신 lablup과 늦은시간까지도 QnA를 해주신 담당자분들께 감사를 표합니다.

</p>
</details>

# TAC-GAN-eCommerce
[updated]
* amazon metadata parser and dataloader
* sentencepiece indexer and doc2vec
* text shuffling for data augmentation
* bce loss => lsgan loss

## Text to Image Synthesis
<img width="981" alt="2019-02-28 9 03 57" src="https://user-images.githubusercontent.com/26558158/53531856-d7b95c00-3b37-11e9-9c21-ccb75300cdf6.png">

## Amazon eCommerce Dataset
Amazon Product Metadata: http://snap.stanford.edu/data/amazon/productGraph/
```
{
  "asin": "0000031852",
  "title": "Girls Ballet Tutu Zebra Hot Pink",
  "price": 3.17,
  "imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
  "related":
  {
    "also_bought": ["B00JHONN1S", "B002BZX8Z6", "B00D2K1M3O", "0000031909", "B00613WDTQ", "B00D0WDS9A", "B00D0GCI8S", "0000031895", "B003AVKOP2", "B003AVEU6G", "B003IEDM9Q", "B002R0FA24", "B00D23MC6W", "B00D2K0PA0", "B00538F5OK", "B00CEV86I6", "B002R0FABA", "B00D10CLVW", "B003AVNY6I", "B002GZGI4E", "B001T9NUFS", "B002R0F7FE", "B00E1YRI4C", "B008UBQZKU", "B00D103F8U", "B007R2RM8W"],
    "also_viewed": ["B002BZX8Z6", "B00JHONN1S", "B008F0SU0Y", "B00D23MC6W", "B00AFDOPDA", "B00E1YRI4C", "B002GZGI4E", "B003AVKOP2", "B00D9C1WBM", "B00CEV8366", "B00CEUX0D8", "B0079ME3KU", "B00CEUWY8K", "B004FOEEHC", "0000031895", "B00BC4GY9Y", "B003XRKA7A", "B00K18LKX2", "B00EM7KAG6", "B00AMQ17JA", "B00D9C32NI", "B002C3Y6WG", "B00JLL4L5Y", "B003AVNY6I", "B008UBQZKU", "B00D0WDS9A", "B00613WDTQ", "B00538F5OK", "B005C4Y4F6", "B004LHZ1NY", "B00CPHX76U", "B00CEUWUZC", "B00IJVASUE", "B00GOR07RE", "B00J2GTM0W", "B00JHNSNSM", "B003IEDM9Q", "B00CYBU84G", "B008VV8NSQ", "B00CYBULSO", "B00I2UHSZA", "B005F50FXC", "B007LCQI3S", "B00DP68AVW", "B009RXWNSI", "B003AVEU6G", "B00HSOJB9M", "B00EHAGZNA", "B0046W9T8C", "B00E79VW6Q", "B00D10CLVW", "B00B0AVO54", "B00E95LC8Q", "B00GOR92SO", "B007ZN5Y56", "B00AL2569W", "B00B608000", "B008F0SMUC", "B00BFXLZ8M"],
    "bought_together": ["B002BZX8Z6"]
  },
  "salesRank": {"Toys & Games": 211836},
  "brand": "Coxlures",
  "categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
}
```
## Text2Image Examples
### Guitar category
#### Generated image examples
<img width="933" alt="2019-02-27 2 15 59" src="https://user-images.githubusercontent.com/26558158/53467593-40ea9200-3a9a-11e9-9cf6-1f66da138c88.png">

#### shape variation

1) guitar + acoustics
<img width="455" alt="acoustics text variation" src="https://user-images.githubusercontent.com/26558158/53462629-d4669780-3a87-11e9-9676-fe57fc8856ae.png">

2) guitar + electric
<img width="465" alt="electric shape variation" src="https://user-images.githubusercontent.com/26558158/53462736-37582e80-3a88-11e9-864e-f0de6fea58a4.png">  

#### color variation  
1) guitar + color
<img width="456" alt="color text variation" src="https://user-images.githubusercontent.com/26558158/53462759-48a13b00-3a88-11e9-92ad-cd6db7d274fa.png"> 

## Demo application
1) generating product image from product metadata (product name, category, brand, color) 

[![Watch the demo video](https://img.youtube.com/vi/q-HZAPw6G0o/0.jpg)](https://www.youtube.com/watch?v=q-HZAPw6G0o)
##### ↑ Click this video

2) classify abused product image using discriminator

<img width="1102" alt="2019-02-28 3 03 30" src="https://user-images.githubusercontent.com/26558158/53544627-11a35600-3b6a-11e9-9fb0-ce58750febd2.png">

## Benchmark
### 1070 vs DGX(parallel)

<img width="990" alt="2019-02-28 8 26 20" src="https://user-images.githubusercontent.com/26558158/53530262-a1c5a900-3b32-11e9-89b2-927501fd418b.png">
1070에 비해 3배이상의 속도 차이가 나고 gpu수가 늘어날수록 증가하나 벤치 모델 사이즈가 크지않아 작은 배치에서는 dgx를 full-load 시키지 못하였다. 큰 배치에서는 io가 병목으로 보임

## Reference
[TAC-GAN PAPER](https://arxiv.org/abs/1703.06412)  
[TAC-GAN tensorflow code](https://github.com/dashayushman/TAC-GAN)  
[TAC-GAN pytorch code](https://github.com/neuperc/TAC-GAN_JeHaYaFa)  
[BACKEND AI repo](https://github.com/lablup/backend.ai)
