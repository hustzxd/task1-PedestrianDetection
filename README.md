---


---

<h2 id="pedestrian-detection">Pedestrian detection</h2>
<blockquote>
<p>Video processing analysis assignment.</p>
</blockquote>
<blockquote>
<p>SSD for pedestrian detection in RAP dataset.</p>
</blockquote>
<h3 id="train">Train</h3>
<ul>
<li>
<p>First download the fc-reduced VGG-16 PyTorch base network weights and trained weights at <a href="https://pan.baidu.com/s/1kgyaVRlt6Fch4nScSsk1BA">BaiduYun</a></p>
</li>
<li>
<p>By default, we assume you have downloaded the file in the ssd.pytorch/weights dir:</p>
</li>
</ul>
<pre class=" language-bash"><code class="prism  language-bash"><span class="token function">cd</span> ssd.pytorch
<span class="token comment"># train SSD300 on MS COCO person dataset</span>
./train_coco_person.sh
<span class="token comment"># train SSD300 on Pascal VOC 2007+2012 dataset</span>
./train_voc_person.sh
<span class="token comment"># train SSD300 on both COCO and VOC dataset</span>
./train_voc_coco_perosn.sh
</code></pre>
<h3 id="evaluation">Evaluation</h3>
<pre class=" language-bash"><code class="prism  language-bash"><span class="token function">cd</span> ssd.pytorch
<span class="token comment"># eval on Pascal VOC 2007</span>
./eval_voc_person.sh weights/<span class="token operator">&lt;</span>you.pth<span class="token operator">&gt;</span>

<span class="token comment"># eval on RAP test dataset</span>
./test_rap.sh weights/<span class="token operator">&lt;</span>you.pth<span class="token operator">&gt;</span>
</code></pre>
<h3 id="performace">Performace</h3>
<p>

We just evaluation AP of person in Pascal VOC 2007.</p>

<table>
<thead>
<tr>
<th align="center">Training Data</th>
<th align="center">Original</th>
<th align="center">Only person(this project)</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">07+12</td>
<td align="center">76.2 %</td>
<td align="center">77.8%</td>
</tr>
<tr>
<td align="center">07+12+COCO</td>
<td align="center">81.4%</td>
<td align="center">82.98%</td>
</tr>
<tr>
<td align="center">07+12+COCO+RAP</td>
<td align="center">—</td>
<td align="center">82.49%</td>
</tr>
</tbody>
</table><h2 id="reference">Reference</h2>
<ul>
<li>Wei Liu, et al. “SSD: Single Shot MultiBox Detector.” <a href="(http://arxiv.org/abs/1512.02325)">ECCV2016</a>.</li>
<li>Thanks to <a href="https://github.com/amdegroot/ssd.pytorch">amdegroot</a>.</li>
</ul>

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI1ODYwODYwNl19
-->