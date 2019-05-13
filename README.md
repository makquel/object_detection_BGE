# Detector de marcadores para cirugia pediatrica
A  anorretoplastia sagital posterior (PSARP-Posterior sagittal anorectoplasty) é uma técnica de reparo de anomalias anorretais (AAR) baseada na exposição completa da região anorretal. Esta cirurgia visa o estabelecimento do controle do esfíncter fecal numa malformação denominada ânus imperfurado alto. O índice de ocorrência mostra que uma, entre 1:1500 a 1:5000 crianças recém-nascidas, apresenta essa anomalia. Foi aprendido da técnica, desenvolvida pelo Professor Doutor Alberto Peña (Journal of pediatry surgery, 1982), que o esfíncter externo é uma estrutura proeminente funcionalmente útil. A identificação desta estrutura muscular teve um papel importante na execução do procedimento, e para este fim a BGE desenvolveu em 2005 um estimulador muscular que permite, ao cirurgião, realizar uma incisão precisa sem danificar o a estrutura muscular do esfíncter, tornando-se a única empresa brasileira em fabricar este tipo de equipamento hospitalar.

O objetivo principal a modernização do estimulador muscular EM901 (estimulador desenvolvido pela BGE). Isto compreende a utilização de técnicas de visão computacional visando assistir o cirurgião durante as fases de estimulação muscular no decorrer do procedimento. Diversos algoritmos de visão computacional foram testados para auxiliar na fase da indentificação muscular, no entanto a solução mais robusta está baseada na utilização de Redes neurais convolucionais para a identifação de marcadores sinteticos na pele.

[![YouTube](https://github.com/makquel/object_detection_BGE/blob/master/sandbox/results/bbosex_filtered.png)](https://www.youtube.com/watch?v=Nnkn2jTRf_I&feature=youtu.be)

## Getting Started ##
Os scripts foram testados no Ubuntu 18.04

### Pré-requisito
* pyhton 3.x
* OpenCV 3.2.0
* Tensoflow 1.12.0

### Treinando a CNN para reconhecimento de marcadores sinteticos
Faça um cópia do jupyter notebook tensorflow_object_detection_training_colab.ipynb no seu google Drive para treinamento no Colab.
Assim que atingir os críterios de parada do treinamento, exporte o modelo de inferência (Frozen network)
Chame o modelo de inferência no script ASARP e execute-o utilizando como entrada um vídeo do procedimento

```
python ASARP_detection_KF.py --video ./video/bge_teste.avi
```
### Sandbox ###
Antes de chegar numa solução parcial ao problema de identificação do movimento muscular através de marcadores sinteticos, foram aboardados outros algoritmos baseados em técnicas de visão computacional. A pasta sandbox contém os scripts implementados para tal.

#### Amplifição de vídeo Euleriana
O resultado do algoritmo [here](https://lambda.qrilab.com/site/tag/tag-science/#show-results)

#### Fluxo óptico
```
jupyter nbconvert --to notebook --execute mynotebook.ipynb --output fluxo_optico.ipynb
```
#### Estimação de Homografia
```
jupyter nbconvert --to notebook --execute mynotebook.ipynb --output Homografy_Trans.ipynb
```
#### Segmentação no espaço de cor
```
jupyter nbconvert --to notebook --execute mynotebook.ipynb --output Hcolor_space.ipynb
```
#### Utilidades
```
jupyter nbconvert --to notebook --execute mynotebook.ipynb --output bge_utilities.ipynb
```




