<p align="center">
  <h1 align="center">DGA: Ataque Guiado por Dirección contra la Detección Óptica Aérea en Escenarios Independientes de la Dirección de Disparo de la Cámara</h1>
  <p align="center">
    <b> TGRS, 2024 </b>
    <!-- <br /> -->
    <a href="https://zytx121.github.io/"><strong>Yue Zhou </strong></a>
    .
    <strong> Shuqi Sun </strong>
    .
    <a href="https://ee.sjtu.edu.cn/FacultyDetail.aspx?id=53&infoid=66&flag=66"><strong> Xue Jiang </strong></a>
    .
    <strong> Guozheng Xu </strong>
    .
    <strong> Fengyuan Hu </strong>
    .
    <strong> Ze Zhang </strong>
    .
    <a href="https://www.researchgate.net/profile/Xingzhao-Liu"><strong> Xingzhao Liu </strong></a>
    .
  </p>

  <p align="center">
    <a href='https://zytx121.github.io/projects/dga'>
      <img src='https://img.shields.io/badge/Project-Link-green?style=flat&logo=GITHUB&logoColor=black' alt='Project'>
    </a>
    <a href='https://ieeexplore.ieee.org/document/10497142'>
      <img src='https://img.shields.io/badge/TGRS-PDF-blue?style=flat&logo=IEEE&logoColor=green' alt='TGRS PDF'>
    </a>
  </p>
<br />


Los ataques adversariales basados en parches han generado cada vez más preocupación debido a su potencial de aplicación en campos militares y civiles. En imágenes aéreas, numerosos objetivos presentan una direccionalidad inherente, como vehículos y barcos, lo que da lugar a la aparición de tareas de detección de objetos orientados; de manera similar, los parches adversariales también exhiben una orientación intrínseca debido a su falta de simetría perfecta. Los métodos existentes presuponguen una alineación estática entre la orientación del parche adversarial y el sistema de coordenadas de la cámara, una suposición que con frecuencia se viola en imágenes aéreas, cuya eficacia se degrada en escenarios del mundo real. En este artículo, investigamos el aspecto a menudo pasado por alto de la orientación del parche en los ataques adversariales y su impacto en la efectividad del camuflaje, particularmente cuando la orientación no es congruente con el objetivo. Se propone un nuevo marco de Ataque Guiado por Dirección (DGA, por sus siglas en inglés) para engañar a los detectores aéreos del mundo real, que muestra un rendimiento de ataque robusto y adaptable en escenarios **independientes de la dirección de disparo de la cámara (CSDA)**. La idea central de DGA es utilizar transformaciones afines para restringir la orientación relativa del parche respecto al objetivo e introducir tres tipos de pérdida para reducir la confianza en la detección del objetivo, hacer que el color sea imprimible y suavizar el color del parche. Introducimos una metodología de evaluación guiada por dirección para tender un puente entre el rendimiento del parche en el dominio digital y su eficacia real en el mundo real. Además, establecemos **un conjunto de datos de detección de vehículos basado en drones (SJTU-4K)**, que etiqueta la orientación del objetivo, para evaluar la robustez de los parches bajo diversas alturas de disparo y vistas. Se realizan extensos experimentos proporcionalmente escalados y 1:1 en escenarios físicos, demostrando la superioridad y el potencial del marco propuesto para ataques en el mundo real.


## Requisitos

```
mmengine == 0.6.0
mmcv == 2.0.0rc4
mmdet == 3.0.0rc6
mmyolo == 3.0.0rc6
mmrotate == 1.0.0rc1
```

## Instalación

```bash
conda create -n camors python=3.8 -y
conda activate camors

conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

pip install -U openmim 
pip install yapf==0.40.1
mim install mmengine==0.6.0
mim install "mmcv==2.0.0rc4" 

cd mmdetection
pip install -v -e .

cd ../mmyolo
pip install -v -e .
pip install albumentations

cd ../mmrotate
pip install -v -e .

```


## Ataques Físicos

Los ataques contra redes neuronales a menudo se realizan mediante el uso de pegatinas o parches, donde el color de los parches o pegatinas no necesita parecerse al fondo. Sin embargo, el área cubierta por las pegatinas no debe ser demasiado grande, con la perturbación caracterizada por un alcance estrecho y una perturbación significativa.

Plan de Métodos:

- :heavy_check_mark: [DPatch](https://github.com/veralauee/DPatch) (AAAIW'2019)
- :heavy_check_mark: [OBJ](https://gitlab.com/EAVISE/adversarial-yolo) (2019)
- :heavy_check_mark: [APPA](https://ieeexplore.ieee.org/abstract/document/9965436) (TGRS'2022)
- :heavy_check_mark: [DGA](https://ieeexplore.ieee.org/abstract/document/) (TGRS'2024)
- :clock3: [Patch-Noobj](https://www.mdpi.com/2072-4292/13/20/4078) (RS'2021)
- :clock3: [APA](https://www.mdpi.com/2072-4292/14/21/5298) (RS'2022)
- :heavy_plus_sign: [APC](https://repository.uantwerpen.be/docman/irua/16bd0a/p177670.pdf)  (2020)
- :heavy_plus_sign: [AerialAttack](https://openaccess.thecvf.com/content/WACV2022/html/Du_Physical_Adversarial_Attacks_on_an_Aerial_Imagery_Object_Detector_WACV_2022_paper.html?ref=https://githubhelp.com) (WACV'2022)
- :heavy_plus_sign: [AdvSticker](https://github.com/jinyugy21/Adv-Stickers_RHDE) (TPAMI'2022)
- :heavy_plus_sign: [SOPP](https://github.com/shighghyujie/newpatch-rl) (TPAMI'2022)
- :heavy_plus_sign: [Adversarial Defense in Aerial Detection](https://robustart.github.io/long_paper/08.pdf)

Entrenamiento del modelo detector:

```bash
python tools/train.py projects/camors/configs/yolov5_s-v61_syncbn_1xb2-100e_sjtu-1024.py
```

Entrenamiento del parche adversarial (Asegúrese de estar en el directorio `mmrotate`):

```bash
python tools/train.py projects/camors/configs/dga/dga_yolov5_s-v61_syncbn_1xb2-5e_sjtu-1024.py
```

Prueba del parche adversarial:

1. Descomente `patch_dir` en `dga_yolov5_s-v61_syncbn_1xb2-5e_sjtu-1024.py`, ya que se utiliza para especificar el parche.
2. Ejecute
```bash
python tools/test.py projects/camors/configs/dga/dga_yolov5_s-v61_syncbn_1xb2-5e_sjtu-1024.py \
    work_dirs/yolov5_s-v61_syncbn_1xb2-100e_sjtu-1024/epoch_100.pth
```


## Conjunto de Datos SJTU-4K

<img width="489" alt="image" src="https://github.com/zytx121/DGA/assets/10410257/5711a95a-f33a-4724-bc2b-2a120da3f695">

<img width="489" alt="image" src="https://github.com/zytx121/DGA/assets/10410257/6a07b3a2-80c1-4434-a5c8-581e1965343b">


Para recopilar los datos que necesitamos, diseñamos un esquema razonable para la captura de datos. Primero, elegimos 20 escenas en nuestro campus como nuestro sitio experimental, incluyendo calles y estacionamientos, donde se ven muchos tipos de automóviles. Posteriormente, utilizamos el dron DJI Mini 3 como herramienta de captura. En nuestro esquema, la altura de vuelo varía de 20 m a 120 m, habiendo un total de 9 alturas de vuelo. La resolución de las imágenes originales es de 4000 $\times$ 2260 píxeles. También proporcionamos imágenes aéreas capturadas en diferentes ángulos de inclinación y direcciones, que se aproximan más a las escenas reales. La Figura 1 muestra varias imágenes capturadas por nuestro dron con diferentes ángulos de inclinación desde las alturas de 20 m, 40 m, 80 m y 110 m, respectivamente. El tamaño de los objetos es bastante diverso, y la cantidad de objetos en una sola imagen es grande, especialmente para las imágenes capturadas a gran altura. La distribución detallada de los datos en el conjunto de datos SJTU-4K se muestra en la Figura 2. Sin embargo, este tamaño es demasiado grande, siendo inadecuado para nuestro experimento. Por lo tanto, necesitamos realizar algún procesamiento de las imágenes originales. Se adaptan a un tamaño más pequeño de 1024 $\times$ 1024 cuando se atacan. En última instancia, hay 10260 imágenes de entrenamiento con 57172 objetos de automóviles y 6345 imágenes de prueba con 19969 objetos de automóviles. Para cooperar con el nuevo método de evaluación de ataques basado en parches propuesto, utilizamos cajas rotadas para etiquetar los vehículos. A diferencia de la representación común de caja rotada, utilizamos una representación de ángulo de 360 grados para conservar la información de dirección de los vehículos. 

El conjunto de datos ya está disponible. 

[GoogleDrive](https://drive.google.com/drive/folders/1LWXC-a7OM2kGbXeCp-frrMcBOFD127Hb?usp=sharing)

[jbox](https://jbox.sjtu.edu.cn/l/j1vS3y)(Código: jlfr)

## Referencias

1. [APPA](https://github.com/JiaweiLian/AP-PA)
2. [mmrotate](https://github.com/open-mmlab/mmrotate)
3. [mmyolo](https://github.com/open-mmlab/mmyolo)
4. [mmdetection](https://github.com/open-mmlab/mmdetection)


## Cita

Si utiliza este proyecto para ataques en su investigación, considere citar

```
@article{zhou2024dga,
  title={DGA: Direction-Guided Attack Against Optical Aerial Detection in Camera Shooting Direction-Agnostic Scenarios},
  author={Zhou, Yue and Sun, Shuqi and Jiang, Xue and Xu, Guozheng and Hu, Fengyuan and Zhang, Ze and Liu, Xingzhao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--22},
  year={2024}
}


@article{zhou2023camonet,
  title={CamoNet: A Target Camouflage Network for Remote Sensing Images Based on Adversarial Attack},
  author={Zhou, Yue and Jiang, Wanghan and Jiang, Xue and Chen, Lin and Liu, Xingzhao},
  journal={Remote Sensing},
  volume={15},
  number={21},
  year={2023}
}

```
