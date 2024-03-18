# 1 反推提示词

![img](imgs/0-finetune/83XRUGMiTXw-1688188918827-3.jpg)

## 1.1 BLIP



## 1.2 CLIP



## 1.2 DeepBooru

**DeepDanbooru**

- [`deepdanbooru-v3-20211112-sgd-e28`](https://github.com/KichangKim/DeepDanbooru/releases/tag/v3-20211112-sgd-e28) ：

  ```text
  1girl, animal ears, cat ears, cat tail, clothes writing, full body, rating:safe, shiba inu, shirt, shoes, simple background, sneakers, socks, solo, standing, t-shirt, tail, white background, white shirt
  ```

- [`deepdanbooru-v4-20200814-sgd-e30`](https://github.com/KichangKim/DeepDanbooru/releases/tag/v4-20200814-sgd-e30)

  ```
  1girl, animal, animal ears, bottomless, clothes writing, full body, rating:safe, shirt, shoes, short sleeves, sneakers, solo, standing, t-shirt, tail, white background, white shirt
  ```

## 1.3 Tagger

链接：https://github.com/toriato/stable-diffusion-webui-wd14-tagger

- 不是一个独立的模型，而是集成了各种其他模型，如DeepBanbooru，wd14-vit-v2 等进行反推
- 输出是 booru 风格的 tag，而不是句子描述

各种反推模型的对比如下：

**DeepDanbooru**

- [`deepdanbooru-v3-20211112-sgd-e28`](https://github.com/KichangKim/DeepDanbooru/releases/tag/v3-20211112-sgd-e28) ：

  ```text
  1girl, animal ears, cat ears, cat tail, clothes writing, full body, rating:safe, shiba inu, shirt, shoes, simple background, sneakers, socks, solo, standing, t-shirt, tail, white background, white shirt
  ```

- [`deepdanbooru-v4-20200814-sgd-e30`](https://github.com/KichangKim/DeepDanbooru/releases/tag/v4-20200814-sgd-e30)

  ```
  1girl, animal, animal ears, bottomless, clothes writing, full body, rating:safe, shirt, shoes, short sleeves, sneakers, solo, standing, t-shirt, tail, white background, white shirt
  ```

**Waifu Diffusion Tagger**

- [`wd14-vit`](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger)

  ```
  1boy, animal ears, dog, furry, leg hair, male focus, shirt, shoes, simple background, socks, solo, tail, white background
  ```

- [`wd14-convnext`](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger)

  ```
  full body, furry, shirt, shoes, simple background, socks, solo, tail, white background
  ```

- [`wd14-vit-v2`](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)

  - 使用 https://github.com/SmilingWolf/SW-CV-ModelZoo 进行训练

  ```
  1boy, animal ears, cat, furry, male focus, shirt, shoes, simple background, socks, solo, tail, white background
  ```

- [`wd14-convnext-v2`](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)

  ```
  animal focus, clothes writing, earrings, full body, meme, shirt, shoes, simple background, socks, solo, sweat, tail, white background, white shirt
  ```

- [`wd14-swinv2-v2`](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2)

  ```
  1boy, arm hair, black footwear, cat, dirty, full body, furry, leg hair, male focus, shirt, shoes, simple background, socks, solo, standing, tail, white background, white shirt
  ```


## 1.4 CLIP-Interrogator

https://github.com/pharmapsychotic/clip-interrogator

可自定义：

### 1.4.1 图像网站

可以判断图像是从哪个网站上下载的，或与哪个网站的风格比较接近。现有网站包括：

```
Artstation
behance
cg society
cgsociety
deviantart
dribbble
flickr
instagram
pexels
pinterest
pixabay
pixiv
polycount
reddit
shutterstock
tumblr
unsplash
zbrush central
```

在上述网站名称的基础上，还进行了扩展，完整代码如下：

```python
sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribbble', 
                 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 
                 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
trending_list = [site for site in sites]
trending_list.extend(["trending on "+site for site in sites])
trending_list.extend(["featured on "+site for site in sites])
trending_list.extend([site+" contest winner" for site in sites])
```

### 1.4.2 宏观风格

提供的风格包括，对应于 抽象主义，抽象表现主义， 魔幻现实主义等等 ：

```
abstract art
abstract expressionism
abstract illusionism
academic art
action painting
aestheticism
afrofuturism
altermodern
american barbizon school
american impressionism
american realism
american romanticism
american scene painting
analytical art
antipodeans
arabesque
arbeitsrat für kunst
art & language
art brut
art deco
art informel
art nouveau
art photography
arte povera
arts and crafts movement
ascii art
ashcan school
assemblage
australian tonalism
auto-destructive art
barbizon school
baroque
bauhaus
bengal school of art
berlin secession
black arts movement
brutalism
classical realism
cloisonnism
cobra
color field
computer art
conceptual art
concrete art
constructivism
context art
crayon art
crystal cubism
cubism
cubo-futurism
cynical realism
dada
danube school
dau-al-set
de stijl
deconstructivism
digital art
ecological art
environmental art
excessivism
expressionism
fantastic realism
fantasy art
fauvism
feminist art
figuration libre
figurative art
figurativism
fine art
fluxus
folk art
funk art
furry art
futurism
generative art
geometric abstract art
german romanticism
gothic art
graffiti
gutai group
happening
harlem renaissance
heidelberg school
holography
hudson river school
hurufiyya
hypermodernism
hyperrealism
impressionism
incoherents
institutional critique
interactive art
international gothic
international typographic style
kinetic art
kinetic pointillism
kitsch movement
land art
les automatistes
les nabis
letterism
light and space
lowbrow
lyco art
lyrical abstraction
magic realism
magical realism
mail art
mannerism
massurrealism
maximalism
metaphysical painting
mingei
minimalism
modern european ink painting
modernism
modular constructivism
naive art
naturalism
neo-dada
neo-expressionism
neo-fauvism
neo-figurative
neo-primitivism
neo-romanticism
neoclassicism
neogeo
neoism
neoplasticism
net art
new objectivity
new sculpture
northwest school
nuclear art
objective abstraction
op art
optical illusion
orphism
panfuturism
paris school
photorealism
pixel art
plasticien
plein air
pointillism
pop art
pop surrealism
post-impressionism
postminimalism
pre-raphaelitism
precisionism
primitivism
private press
process art
psychedelic art
purism
qajar art
quito school
rasquache
rayonism
realism
regionalism
remodernism
renaissance
retrofuturism
rococo
romanesque
romanticism
samikshavad
serial art
shin hanga
shock art
socialist realism
sots art
space art
street art
stuckism
sumatraism
superflat
suprematism
surrealism
symbolism
synchromism
synthetism
sōsaku hanga
tachisme
temporary art
tonalism
toyism
transgressive art
ukiyo-e
underground comix
unilalianism
vancouver school
vanitas
verdadism
video art
viennese actionism
visual art
vorticism
```

### 1.4.3 艺术形式

风格主义的标签是 xxx主义，风格形式会更加具体，如3D 渲染，一张黑白照片，卡通，水粉画，详细的哑光绘画 等等，具体包含：

```
a 3D render
a black and white photo
a bronze sculpture
a cartoon
a cave painting
a character portrait
a charcoal drawing
a child's drawing
a color pencil sketch
a colorized photo
a comic book panel
a computer rendering
a cross stitch
a cubist painting
a detailed drawing
a detailed matte painting
a detailed painting
a diagram
a digital painting
a digital rendering
a drawing
a fine art painting
a flemish Baroque
a gouache
a hologram
a hyperrealistic painting
a jigsaw puzzle
a low poly render
a macro photograph
a manga drawing
a marble sculpture
a matte painting
a microscopic photo
a mid-nineteenth century engraving
a minimalist painting
a mosaic
a painting
a pastel
a pencil sketch
a photo
a photocopy
a photorealistic painting
a picture
a pointillism painting
a polaroid photo
a pop art painting
a portrait
a poster
a raytraced image
a renaissance painting
a screenprint
a screenshot
a silk screen
a sketch
a statue
a still life
a stipple
a stock photo
a storybook illustration
a surrealist painting
a surrealist sculpture
a tattoo
a tilt shift photo
a watercolor painting
a wireframe diagram
a woodcut
an abstract drawing
an abstract painting
an abstract sculpture
an acrylic painting
an airbrush painting
an album cover
an ambient occlusion render
an anime drawing
an art deco painting
an art deco sculpture
an engraving
an etching
an illustration of
an impressionist painting
an ink drawing
an oil on canvas painting
an oil painting
an ultrafine detailed painting
chalk art
computer graphics
concept art
cyberpunk art
digital art
egyptian art
graffiti art
lineart
pixel art
poster art
vector art
```

### 1.4.4 正向提示词

正向提示词是一个大词表，即包含dog, cat等简单提示词，也包含highly detailed, 4k, 8k, digital painting, 辛烷渲染等提示词，更符合目前AIGC的prompt形式。共提供10w+提示词，具体见flavors.txt。

需要注意，10w+提示词可能没有经过人工清洗，会包含 `8 k` ，`8 k - h 7 6 8` 等不规范提示词，需要清洗。





## 1.5 BLIP2

https://github.com/salesforce/LAVIS/tree/main/projects/blip2

## 1.6 生成朋友圈

https://pallyy.com/tools/image-caption-generator



# 1.7 开源网站提供了多种示例

https://replicate.com/collections/image-to-text





- ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
- ci.interrogate(image)

```
there is a little girl that is standing in a field of flowers, beautiful gorgeous digital art, inspired by Anne Geddes, lorem ipsum dolor sit amet, cute young man, gentle face, holding intimately, arts, colored vibrantly, profile picture 1024px, richly colored


there is a little girl with a big afro with a bunch of balls, substance 3d painter, trending on artstration, style of maple story, dress in the style of rococo, by Abdullah Gërguri, very cute features, well rendered.:1, clay render, big hair, cartoonish cute

```

- ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
- ci.interrogate_fast(image)

```
there is a little girl that is standing in a field of flowers, very beautiful digital art, beautiful gorgeous digital art, beautiful portrait image, realistic colorful photography, beautiful digital artwork, beautiful art uhd 4 k, beautiful digital painting, exquisite digital art, beautiful digital art, adorable digital painting, beautiful!!! digital art, flowers and butterflies, by Pamela Ascherson

there is a little girl with a big afro with a bunch of balls, cute! c4d, cute 3 d render, 3 d character art, realistic anime 3 d style, small character. unreal engine 5, 3d character, 3 d character, animated character design, render of a cute 3d anime girl, 3 d render character art 8 k
```

- ci = Interrogator(Config(clip_model_name="ViT-B-32/openai"))
- ci.interrogate_fast(image)





```
there is a little girl with a big afro with a bunch of balls, cute! c4d, cute 3 d render, 3 d character art, realistic anime 3 d style, small character. unreal engine 5, 3d character, 3 d character, animated character design, render of a cute 3d anime girl, 3 d render character art 8 k
```

![16950699593935096_0_0](imgs/0-tagger/16950699593935096_0_0.png)





```
there is a little girl that is standing in a field of flowers, very beautiful digital art, beautiful gorgeous digital art, beautiful portrait image, realistic colorful photography, beautiful digital artwork, beautiful art uhd 4 k, beautiful digital painting, exquisite digital art, beautiful digital art, adorable digital painting, beautiful!!! digital art, flowers and butterflies, by Pamela Ascherson
```

![007294c6-872a-4cd9-b8be-5e411fe7e13d](imgs/0-tagger/007294c6-872a-4cd9-b8be-5e411fe7e13d.jpg)
