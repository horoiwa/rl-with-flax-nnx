# DQN Atari Breakout

このプロジェクトは、深層強化学習（Deep Reinforcement Learning）の一手法であるDQN（Deep Q-Network）を用いて、AtariのBreakout（ブロック崩し）を攻略するエージェントを学習させるものです。

## 必要なもの

*   Python 3
*   [uv](https://docs.astral.sh/uv/getting-started/installation/) (Pythonのパッケージ管理ツール)

`uv` がインストールされていない場合は、上記のリンク先を参照してインストールしてください。

## セットアップ

1.  **リポジトリのクローン:**
    まず、このプロジェクトをローカルマシンにクローンします。

2.  **ディレクトリの移動:**
    クローンしたリポジトリ内の `dqn_atari_breakout` ディレクトリに移動します。
    ```bash
    cd dqn_atari_breakout
    ```

3.  **依存関係のインストール:**
    `uv` を使って、プロジェクトに必要なPythonパッケージをインストールします。
    ```bash
    uv sync
    ```

## トレーニングの実行

以下のコマンドで、DQNエージェントのトレーニングを開始します。

```bash
uv run python manage.py --use-wandb
```

`--use-wandb` フラグを付けることで、学習の進捗が [Weights & Biases](https://wandb.ai/) に記録されます。

### **注意：エラーが発生した場合**

もし `"opencv-python package not installed..."` のようなエラーが表示された場合は、Atari環境の描画に必要なライブラリが不足しています。

以下のコマンドで `libgl1` をインストールしてください。

```bash
sudo apt update && sudo apt install -y libgl1
```

その後、再度トレーニングの実行コマンドをお試しください。

## 参考資料

*   **Flax NNX:** [Flax NNX Documentation](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html)
*   **DQN (原論文):** [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
*   **Gymnasium:** [Gymnasium Documentation](https://gymnasium.farama.org/)
*   **Arcade Learning Environment (ALE):** [ALE Documentation](https://ale.farama.org/)