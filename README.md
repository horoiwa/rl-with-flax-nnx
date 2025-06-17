# Reinforcement-learning-with-flax-nnx

これはFlax NNXによる代表的な深層強化学習アルゴリズムのサンプル実装集です。理解しやすさのため、DRY原則にもとづくコンポーネントの共通化は最低限にとどめ、各アルゴリズムの実装が可能な限り独立になるようにしています。

すべての実装はALE/BreakOutにより妥当なパフォーマンスを発揮することを検証しています。

注意：これはScikit-learnなどのような再利用されることを想定したツールキットではありません。


## アルゴリズム

```凡例
### アルゴリズム名
[論文へのリンク](URL)

[Description]
手法の簡単な説明(300字程度)

[Implementation]

検証スクリプトの実行コマンド: `train.sh {dqn/a2c/ppo}`

実装における留意事項（環境や訓練ステップ数、実行時間など）

スコアの遷移（画像）

訓練完了時の動き（gif)


```


### DQN

### A2C

### PPO






## References

[Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html)

[Gymnasium](https://gymnasium.farama.org/)

[ALE](https://ale.farama.org/)
