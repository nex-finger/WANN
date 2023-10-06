# 覚えておくこと
2031133 増田瑞樹

## はじめに
- WANNの研究についての備忘録
- pythonのバージョンとかpipでインストールとかそのレベルはもう書いてません，提案手法についてが主です
- VSCodeで閲覧する際はこのテキストを読み込んで Ctrl + Shift + V でプレビュー
- WANN自体のREADMEはREADME/README.mdにあります

## 目次
[9/22](#20230922)  
[9/27](#20230927)  
[9/29](#20230929)  
[10/3](#20231003)  

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
ルーレット選択をする場合は重みありランダム選択で実装すると楽そう．
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

    - neat_src/_variation.py にて(1)を確認，(2)で neat_src/wann_ind.py の Wann.Ind かららしい（neat_src/ind.pyではない）
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

## 2023/10/03
- 10月になってしまいました
    - 内定式に行ってたら今週も進捗ないしREADMEも吹き飛ぶしおれは終わりです

- デフォルトの使用プロセッサ少なくないですか
    - 実行されているタスクの確認作業中気がついたんだけど多分2コアで動いてる
    ```python
    parser.add_argument('-n', '--num_worker', type=int,\
    help='number of cores to use', default=2)
    ```
    - のでターミナルより
    ```bsh
    lscpu

    CPU:                  
    model name         : 12th Gen Intel(R) Core(TM) i7-12700
    core(s) per socket : 12
    ```
    - を確認して12コア使えるのでプログラムのデフォルト値を12にしてみる
        - エラーでました  
        default=11だとうまく行ったので11コアで今後動かします  
        11倍速には当然ならないけど2倍速以上にはなってる感覚
    - この記述でも動きます
    ```bsh
    python wann_train.py -n 11
    ```
    - 追記  
    なんか表記を見る限りこのパラメータ+1のコアで動いてるみたい  
    だから11で動かせばこのPCで最大の12コアで動くし12にすると13コア存在しないのでエラーがでるみたい  
    以上

- 本当に10世代でBipedタスクでまともに歩けてるのか動画で確認してほしい
    - そもそもBipedが動いてるのか怪しくなってきた  
    スコアが300ちょいだったので勝手に思い込んでて確認してませんでした
    - wann_train.py内にて(またneat_train.pyをいじってて時間を無駄にしました)
    ```python
    parser.add_argument('-p', '--hyperparam', type=str,\
    help='hyperparameter file', default='p/laptop_swing.json')
    ```
    - 明らかにBipedではなくてわろた  
    以下に置き換えpipでBox2Dを入れておきます
    ```python
    parser.add_argument('-p', '--hyperparam', type=str,\
    help='hyperparameter file', default='p/biped.json')
    ```
    - Bipedにしたら普通に時間かかるようになりました
    - 追記  
    デフォルトをlaptop_swingに戻しました  
    理由はたいてい場合こっちが指定したmasudaファイルを使うっていうのとswingのほうが1世代が早くてデバッグに好都合だからです  
    以上
    
- でこっからBipedタスクで歩いている様子を動画にしたいのですが
    - デフォルトのp/biped.jsonでは状態を保存するパラメータが記述されていないのでp/masuda/biped1003.jsonに新しくつくります  
    save_modの項目を追加(8世代ごと)，詳しくはp/hypkey.txtにかいてありますからそのくらいの英語は読んでください
    - p/masuda/biped1003.jsonの内容
    ```json
    {
        "task":"biped",
        "alg_nReps": 4,
        "maxGen": 2048,
        "popSize": 480,
        "prob_initEnable": 0.25,
        "select_tournSize": 16,
        "save_mod" : 8
    }
    ```
    - 実行はこう  
    パラメータが-pで稼働コアが-nですね（今後このレベルの解説は）
    ```bsh
        python wann_train.py -p p/masuda/biped1003.json -n 11
    ```

- 動作をmain関数から一括で変える
    - いちいちプログラムの深くにいって，回したい処理のコメントアウトを消して回したくない処理にコメントアウトをつけるのは情弱なのでやめます  
    jsonファイルに変数を保存してそこからプログラム開始時にグローバル変数として保持しておく
    - jsonファイルは p/masuda/branch.json  
    いまのところ活性化関数の変更基準だけを保存してあるが今後追加できる

    - 活性化関数変更プログラム
    ```python
        # Mutate Activation
        elif choice == 4:
        start = 1+self.nInput + self.nOutput
        end = nodeG.shape[1]           
        if start != end:
            mutNode = np.random.randint(start,end)
            branchData , act_change_alg = masudaSetup()

            #ここからかきました
            print(branchData['act_change_alg'])
            if branchData['act_change_alg'] == 0:     #既存手法
            newActPool = listXor([int(nodeG[2,mutNode])], list(p['ann_actRange']))
            nodeG[2,mutNode] = int(newActPool[np.random.randint(len(newActPool))])

            elif branchData['act_change_alg'] == 1:   #-3から3の積分
            _i = int(nodeG[2,mutNode])[0] -1
            nodeG[2,mutNode] = int(random.choices(activateChangeTable[0], weights = activateChangeTable[1][_i])[0])
            
            elif branchData['act_change_alg'] == 2:   #-2から2の積分
            _i = int(nodeG[2,mutNode])[0] -1
            nodeG[2,mutNode] = int(random.choices(activateChangeTable[0], weights = activateChangeTable[2][_i])[0])

        child = WannInd(connG, nodeG)
        child.birth = gen

        return child, innov

    # -- masuda no jikan ----------------------------------------------------- -- #
    def masudaSetup():
    with open('p/masuda/branch.json', 'r') as json_file:
        data = json.load(json_file)

    table = []
    table.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    table.append([
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
    ]) #分解能0.001，-3から3の積分
    table.append([
        [0.0   , 0.1165, 0.0393, 0.0547, 0.4467, 0.0898, 0.0182, 0.0364, 0.1456, 0.0529],  
        [0.044 , 0.0   , 0.0367, 0.1069, 0.1032, 0.3956, 0.0129, 0.044 , 0.2201, 0.0367], 
        [0.055 , 0.1359, 0.0   , 0.1444, 0.1041, 0.1456, 0.1135, 0.0741, 0.0915, 0.1359],  
        [0.0345, 0.1785, 0.0651, 0.0   , 0.0639, 0.4015, 0.0345, 0.0673, 0.0906, 0.0642],  
        [0.2904, 0.1776, 0.0484, 0.0658, 0.0   , 0.1319, 0.0181, 0.0341, 0.1717, 0.062 ],  
        [0.031 , 0.3621, 0.036 , 0.22  , 0.0701, 0.0   , 0.0152, 0.0522, 0.1719, 0.0415],  
        [0.0481, 0.0905, 0.2142, 0.1445, 0.0736, 0.1163, 0.0   , 0.0961, 0.0769, 0.1398],  
        [0.0485, 0.1551, 0.0705, 0.1421, 0.0698, 0.201 , 0.0485, 0.0   , 0.1939, 0.0706],  
        [0.0744, 0.2978, 0.0334, 0.0734, 0.1349, 0.2542, 0.0149, 0.0744, 0.0   , 0.0425],  
        [0.0703, 0.1289, 0.1289, 0.1351, 0.1266, 0.1592, 0.0703, 0.0703, 0.1105, 0.0   ]
    ]) #分解能0.001，-2から2の積分

    return data, table
    ```

    - で一応0(既存手法)にしたらうまく動いたんだけど1にしたらエラー出ました  
    今日はもう20:30なので次回続き

- githubで管理する
    - 文字通りgithubで管理します  
    パスワード認証が2年くらい前にオワコンと化していたのでSSHキーで認証しました
    - HTTPS https://github.com/nex-finger/WANN
    - SSH git@github.com:nex-finger/WANN.git
    - 今日はpushして帰宅

## 2023/10/6
- jsonファイル0なら動くけど1だと動かないから直していく
    - なぜ動かない  
    ``` bsh
    Traceback (most recent call last):
    File "wann_train.py", line 280, in <module>
        main(args)                              
    File "wann_train.py", line 254, in main
        master()
    File "wann_train.py", line 30, in master
        pop = alg.ask()            # 新しく進化生成した個体集団の取得
    File "/home/masuda/Documents/WANN/WANN/neat_src/neat.py", line 53, in ask
        self.evolvePop()    # Create child population 
    File "/home/masuda/Documents/WANN/WANN/neat_src/_variation.py", line 13, in evolvePop
        children, self.innov = self.recombine(self.species[i],\
    File "/home/masuda/Documents/WANN/WANN/neat_src/_variation.py", line 77, in recombine
        child, innov = pop[parents[0,i]].createChild(p,innov,gen)
    File "/home/masuda/Documents/WANN/WANN/neat_src/wann_ind.py", line 68, in createChild
        child, innov = child.topoMutate(p,innov,gen)
    File "/home/masuda/Documents/WANN/WANN/neat_src/wann_ind.py", line 154, in topoMutate
        _i = int(nodeG[2,mutNode])[0] -1
    TypeError: 'int' object is not subscriptable
    ```

    - 大かっこで囲ったら動いたんだけどなんで大かっこで囲ったら動くのかはわからない
    - あしたもやる気があれば行きます
