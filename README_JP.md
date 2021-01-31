# PyTorch SSD

[English](https://github.com/birosjh/pytorch_ssd/blob/main/README.md)

このプロジェクトはPyTorchのSingle Shot Detector (SSD) の実装です: https://arxiv.org/pdf/1512.02325.pdf

このプロジェクトの目的は私のSSDのコンポーネントの理解を深めることです。その上、SSD論文で書かれている内容を理解して、よくある実装と論文がどのように異なることも確認したいと思っています。コメントは基本的に英語で書くようにしたいと思っていますが、定期的に実装の説明を[Qiita](https://qiita.com/birosjh)で投稿して行きたいと思っています。後は、使用方法や開発の流れをこのREADMEでできるだけ更新します。

私の実装は、この三つのプロジェクトをよく参考にしています:

- https://github.com/NVIDIA/DeepLearningExamples/tree/49e387c788d606f9711d3071961c35092da7ac8c/PyTorch/Detection/SSD
- https://github.com/amdegroot/ssd.pytorch
- https://github.com/kuangliu/pytorch-ssd/blob/master/encoder.py

学習のため、VOC Datasetを利用しています: http://host.robots.ox.ac.uk/pascal/VOC/