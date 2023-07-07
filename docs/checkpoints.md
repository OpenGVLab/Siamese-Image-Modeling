# Checkpoints
We provide links for you to download the checkpoints of SiameseIM models here. 


<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Backbone</th><th>Pretrained Epoch</th><th>Finetuned on ImageNet</th><th>Link</th>
    </tr>
    <tr align="center">
        <td>SiameseIM</td><td>ViT-Base</td><td>1600</td><td>w/o</td><td><a href="">Download</a></td>
    </tr>
     <tr align="center">
        <td>SiameseIM<sub>ft</sub></td><td>ViT-Base</td><td>1600</td><td>w/</td><td><a href="">Download</a></td>
    </tr>
    
</table>

* The SiameseIM model is only pretrained on ImageNet datasets for 1600 epochs. For pretraining details, see [pretrain.md](./pretrain.md).
* The SiameseIM$`_{\mathrm{ft}}`$ model is first pretrained for 1600 epochs, and the finetuned with ImageNet classification task for 100 epochs. For finetuning details, see [finetune.md](./finetune.md).
*  More pre-trained weights will be released.
