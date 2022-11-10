<h1>Text Suicide Detection with NLP</h1>

<h2>Notes :</h2>
<ul>This project was not intended to detect suicide for real - A lot of expertise on psychology and different types of data needs to be considered.</ul>
<ul>Dataset was taken on <a href="https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch" target="_blank">https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch</a></ul>
<ul>Application is deployed on <a href="https://text-suicide-detection.herokuapp.com/" target="_blank">text-suicide-detection.herokuapp.com</a>
<h2>Preprocessing Pipeline :</h2>
<ol>
<li>Stopwords removal.</li>
<li>Stemming and Lemmatizing.</li>
</ol>
<h2>Description</h2>
<h3>Tokenization</h3>
<ul>
<li>Tokenization method : <strong>Word based tokenization</strong></li>
<li>Max Token Sequence : <strong>100</strong></li>
</ul>
<h3>Model</h3>
<ul>
<li>Architecture : <strong><a href="https://arxiv.org/abs/1706.03762" target="_blank">Classic Transformer from Vaswani et. al</a></strong></li>
<li>Parameter : <strong>2,020,621</strong></li>
</ul>
