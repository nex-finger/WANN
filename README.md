# 覚えておくこと
2031133 増田瑞樹

## はじめに
- WANNの研究についての備忘録
- VSCodeで閲覧する際はこのテキストを読み込んで Ctrl + Shift + V でプレビュー
- WANN自体のREADMEはREADME/README.mdにあります

## 2023/09/22
- Bipedタスクについての知識がなさすぎる
    - どんなネットワークなのか
        - https://www.gymlibrary.dev/environments/box2d/bipedal_walker/https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
        - シンプルな四足歩行ロボット環境．  
        わずかに凹凸のある地形と，はしご，切り株，落とし穴のある地形の2種類が用意されているが，多くの場合，前者のタスクが解かれる．

    - スコアの出し方
        - 前に進むと報酬（スコア）を与えられ，地形の一番右端まで到達すると300ポイント以上になる．(+130)  
        ロボットが転ぶと報酬は-100され，スタート地点に戻される．  
        進むために腰や膝のモーターを駆動させる必要があるが，これらにトルクをかけるとわずかに報酬が減る(-0.00035)．  
        胴体の角度が急激に変わると報酬が減る(-5)．
        トルクをかけずにその場に立つことができれば，報酬は減らない（つまり時間経過による報酬の減少は定期用されない）．
        体が地面に接触するか，ロボットが地形の右端に到達するとエピソードは終了する．

    - 今までの精度の良いネットワークのスコアはいくらくらいなのか
        - 素朴な実装でスコア300をわずかに上回る．  
        - https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf を用いたネットワークにてスコア347を実現

    - 入力次元数，出力次元数，それぞれのシナプスの役割
        - 入力は24，胴体の傾きと速度，各関節の角度と角速度，接触判定，LiDAR(レーザー照射による距離の測定，飯会のようなもの)の測定値  

        | 入力シナプス | シナプス出力の値         |
        | ----         | ----                     |
        | state[0]     | 胴体の角度               |
        | state[1]     | 胴体の角速度             |
        | state[2]     | x軸方向の速度            |
        | state[3]     | y軸方向の速度            |
        | state[4]     | 左足付け根の関節の角度   |
        | state[5]     | 左足付け根の関節の角速度 |
        | state[6]     | 左足膝の関節の角度       |
        | state[7]     | 左足膝の関節の角速度     |
        | state[8]     | 左足と地面の接触判定     |
        | state[9]     | 右足付け根の関節の角度   |
        | state[10]    | 右足付け根の関節の角速度 |
        | state[11]    | 右足膝の関節の角度       |
        | state[12]    | 右足膝の関節の角速度     |
        | state[13]    | 右足と地面の接触判定     |
        | state[14-23] | LiDARの測定値            |

- 元論文で書かれている既存手法ってなんですか？
    - よくわかんなかったけど参考文献番号はあった  
    https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf

- 提案手法(2)は一般的に関数の距離として扱っている人はいない
    - 教授からプリントいただきました．  
    距離は最小じゃなくて最大っぽいことを言ってたけどなんかたぶん違う

- 提案手法(3)式変形のミス
    - 普通に次回から訂正して表示する，本質は理解

- 関数の距離を計算したあとに，それらをどのようにしてルーレット選択を行っているのかを説明すること
    - 方法は何であれ，関数の距離を図ったときに，距離が小さいほど評価は高くなるので，距離の逆数を評価値とすること．  
    それらの値をルーレット選択によって選択すること．

## 2023/09/27
- 毎日書こうな
    - はい

- 活性化関数の変更プログラム実装Tips
    - ルーレット選択をする場合は重みありランダム選択で実装すると楽そう．
        - https://3pysci.com/random-2/
        - 基本構文
        ```python
        import random                                               #モジュールインポート

        val_list = [1, 2, 3, 4, 5, 6]                               #選択される要素たちval_list[i]
        weight_list = [10, 1, 1, 1, 1, 1]                           #val_list[i]が選択される確率weight_list[i]

        val = random.choices(val_list, weights=weight_list)[0]      #valに格納，最初の一個だけでいいので[0]
        ```
- 活性化関数を変更しているのはどこですか？
    - 実行は /home/masuda/WANN0712/master/WANNRelease/prettyNeatWann/wann_train.pyより(neat_trainではない！！)
        - wann_train.py で以下を宣言，またalg  = Wann(hyp)
        ```python
        alg = Wann(hyp)
        ```

        - neat_src/neat.py の Class Neat 内に(1)がある，ask 内の(2)で評価の高かった個体の子孫を生成している
        ```pytohn
        def ask(self):
        ```
        ```python
        self.evolvePop()
        ```

        - neat_src/_variation.py にて(1)を確認，(2)で neat_src/wann_ind.py の Wann.Ind かららしい（neat_Sec/ind.pyではない）
        ```python
        def evolvePop(self):
        ``` 
        ```python
        print(type(pop[parents[0,i]]))
        ```

        - すぐ下に topoMutate があり，これが実際に活性化関数を変更しているプログラムになる（ﾅｶﾞ）
        ```python
        # Mutate Activation
        elif choice is 4:
        start = 1+self.nInput + self.nOutput
        end = nodeG.shape[1]           
        if start != end:
            mutNode = np.random.randint(start,end)
            newActPool = listXor([int(nodeG[2,mutNode])], list(p['ann_actRange']))
            nodeG[2,mutNode] = int(newActPool[np.random.randint(len(newActPool))])
        
        #この関数はutils/utils.pyにある
        def listXor(b,c):
        """Returns elements in lists b and c they don't share
        """
        A = [a for a in b+c if (a not in b) or (a not in c)]
        return A
        ```

        - newActPoolはlistXor関数を通して現在の活性化関数から新たしい活性化関数をランダムに選択している

        - utilのbとcをprint文で確認する
        ```python
        print('b')
        print(b)
        print('c')
        print(c)
        print('a')
        print(A)

        #出力(なぜか ID==11 が存在していない)
        #b
        #[7]
        #c
        #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #a
        #[1, 2, 3, 4, 5, 6, 8, 9, 10]
        ```

        - つまり関数の出力はbを除いたcの要素のいずれか（7を除いた1から10までの整数のいすれか）になる

    ## 2023/09/29
    - 興奮のあまり27日の結果を記録してなかったので今書く
        - バカが代
    
    - 活性化関数の変更確率の変化
        - 今2通りの変更を考えていて，ひとつは入力として頻出する範囲-nからnまでの2つの関数の積分(1)と，実際の入力を考慮したときの出力の差(2)
        - $$\int^{n}_{-n} (f(x) - g(x))^{2}$$  
            $$ (f(x_{\sum_{k=1}^{n}k}) - g(x_{\sum_{k=1}^{n}k}))^2$$
        - 以下(1)を「積分の差」，(2)を「特有の差」と記述する
    
    - 積分の差を使った活性化関数変更
        - プログラム変更点  
        ```python
            elif choice == 4:
            start = 1+self.nInput + self.nOutput
            end = nodeG.shape[1]           
            if start != end:
                mutNode = np.random.randint(start,end)

                #ここからかきました
                transRange = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                transList = [
                [0.0   , 0.1294, 0.0904, 0.0785, 0.2763, 0.1142, 0.0216, 0.0431, 0.1726, 0.074 ],
                [0.019 , 0.0   , 0.0484, 0.1006, 0.0956, 0.5955, 0.0076, 0.019 , 0.0762, 0.0381],
                [0.0472, 0.1718, 0.0   , 0.1702, 0.1437, 0.1799, 0.0327, 0.0387, 0.0805, 0.1353],
                [0.0247, 0.2153, 0.1026, 0.0   , 0.0846, 0.3579, 0.0247, 0.0309, 0.0556, 0.1035],
                [0.1049, 0.2466, 0.1044, 0.102 , 0.0   , 0.1935, 0.0154, 0.0268, 0.1224, 0.0841],
                [0.016 , 0.5685, 0.0484, 0.1596, 0.0716, 0.0   , 0.0082, 0.0197, 0.0651, 0.0428],
                [0.0464, 0.1114, 0.1346, 0.1689, 0.087 , 0.1256, 0.0   , 0.0928, 0.0742, 0.1591],
                [0.0512, 0.1537, 0.0878, 0.1166, 0.0838, 0.1662, 0.0512, 0.0   , 0.2049, 0.0845],
                [0.0807, 0.242 , 0.072 , 0.0826, 0.1508, 0.2167, 0.0161, 0.0807, 0.0   , 0.0585],
                [0.0431, 0.1508, 0.1508, 0.1913, 0.1291, 0.1775, 0.0431, 0.0415, 0.0729, 0.0   ],
                ]

                nodeG[2,mutNode] = listXor([int(nodeG[2,mutNode])], transRange, transList)

                """
                newActPool = listXor([int(nodeG[2,mutNode])], list(p['ann_actRange']))
                nodeG[2,mutNode] = int(newActPool[np.random.randint(len(newActPool))])
                """
            
            def listXor(now, state, table):
            """Returns elements in lists b and c they don't share
            """
            i = now[0] -1
            A = int(random.choices(state, weights = table[i])[0])
            
            """
            A = [a for a in b+c if (a not in b) or (a not in c)]
            print('b')
            print(b)
            print('c')
            print(c)
            print('a')
            print(A)
            """
            return A
        ```

        - -3 から 3 までの積分の差を配列に保存して重み付き乱数で選択できるようにしている
        - 結果は従来手法で10世代目Bipedタスクのスコアが352.79に対して，提案手法のスコアが323.30だった（悔しくてもう一回回したら360くらいいったので反復試行の価値あり）
        - ただタスクがかんたんすぎてすぐに最適解に漸近してしまっているのではないかと思ってる
        - って言ったら教授から本当に歩けてるのか動画で確認しろって言われたのでしようと思う
        - もう少し難しいタスクとなると自動車のモデルを運転するものがある，これは最適解に近いドライブはとても難しいと言われている[要出典]
    
    - 中間報告のタイトルについて
        - パっと見でわかるように説明して **「進捗報告」はNG**