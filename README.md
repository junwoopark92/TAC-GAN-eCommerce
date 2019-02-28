# Just Model It
# TAC-GAN-eCommerce
[updated]
* amazon metadata parser and dataloader
* sentencepiece indexer and doc2vec
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

## Demo  
{% include demo_video.html id="q-HZAPw6G0o&feature=youtu.be" %}  

## Benchmark
### 1070 vs DGX(parallel)

<img width="990" alt="2019-02-28 8 26 20" src="https://user-images.githubusercontent.com/26558158/53530262-a1c5a900-3b32-11e9-89b2-927501fd418b.png">
1070에 비해 3배이상의 속도 차이가 나고 gpu수가 늘어날수록 증가하나 벤치 모델 사이즈가 크지않아 작은 배치에서는 dgx를 full-load 시키지 못하였다. 큰 배치에서는 io가 병목으로 보임

## Reference
[TAC-GAN PAPER](https://arxiv.org/abs/1703.06412)  
[TAC-GAN tensorflow code](https://github.com/dashayushman/TAC-GAN)  
[TAC-GAN pytorch code](https://github.com/neuperc/TAC-GAN_JeHaYaFa)  
