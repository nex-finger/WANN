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
[10/6](#20231006)  
[10/9](#20231009)  
[10/10](#20231010)  
[10/11](#20231011)  
[10/13](#20231013)  
[10/16](#20231016)  
[10/17](#20231017)  
[10/20](#20231020)  
[10/24](#20231024)  
[10/26](#20231026)  
[10/31](#20231031)  
[11/2](#20231102)  
[11/3](#20231103)  
[11/7](#20231107)  
[11/10](#20231110)  
[11/16](#20231116)
[11/22](#20231122)

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

## 2023/10/06
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
    あしたもやる気があれば行きます

## 2023/10/09
- どーも結局大学には行かずwindows98をいじっておりました  
今日も頑張りましょう

- シナプス重みを入力としたときの出力の差
    - 式は以下の通り  
    $$ (f(x_{\sum_{k=1}^{n}k}) - g(x_{\sum_{k=1}^{n}k}))^2$$

    - 実際に入力されている値を考慮するため，この手法が一番誤差の少ないものを選んでくれる  
    その上，近傍の値が大きく異なるので有効な探索をしてくれるのではないかと思っている

    - 前回までの実装は，自身の活性化関数IDをテーブルに基づいて変更するだけだったので楽だったが  
    今回は自身の活性化関数に シナプス入力がされているかの情報を持ってこなければならない

- シナプス荷重はどこにあるのですか
    - 予想するに，ネットワークトポロジの変更には活性化関数だけでなくシナプスの追加やノードの追加もある  
    ので，かなり近くにほしいデータは来ていると思う（来ていなかったら5000兆回の配列受け渡しを余儀なくされおれは死ぬ）

    - neat_src/_variation.py にてこんかことが書いてある
    ```
    Creates next generation of child solutions from a species
    Returns:
        children - [Ind]      - newly created population
        innov   - (np_array)  - updated innovation record
    ```
    ```
    種から次世代の子ソリューションを作成します
    戻り値：
       子供 - [Ind] - 新しく作成された人口
       innov - (np_array) - 更新されたイノベーション レコード
    ```

    - 更新されたイノベーションレコードとは多分トポロジを刷新した個体の情報のことでしょう  
    つまり活性化関数だけでなく，どのシナプスがどのノードに接続されているかもわかるはずです

    - _variation.py が呼び出してる neat_src/wann_ind.py 内 recombine にて  
    ```
    Randomly alter topology of individual
    Note: This operator forces precisely ONE topological change 

    Args:
      child    - (Ind) - individual to be mutated
        .conns - (np_array) - connection genes
                  [5 X nUniqueGenes] 
                  [0,:] == Innovation Number (unique Id)
                  [1,:] == Source Node Id
                  [2,:] == Destination Node Id
                  [3,:] == Weight Value
                  [4,:] == Enabled?  
        .nodes - (np_array) - node genes
                  [3 X nUniqueGenes]
                  [0,:] == Node Id
                  [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
                  [2,:] == Activation function (as int)
      innov    - (np_array) - innovation record
                  [5 X nUniqueGenes]
                  [0,:] == Innovation Number
                  [1,:] == Source
                  [2,:] == Destination
                  [3,:] == New Node?
                  [4,:] == Generation evolved

    Returns:
        child   - (Ind)      - newly created individual
        innov   - (np_array) - innovation record
    ```

    - つまり self.conns (多分connectionsの略) の2次元配列は要素数が5つのデータ列になっていて，1行のデータには  
    変更ID，シナプスの伝播元ノードID，シナプスの伝播先ノードID，シナプスの荷重，有効か否かが記されている

- どんな入力（どんな状態）かを知る必要がある
    - ここまできて初めて気づいたんですけど，あるノードの出力を得るためには，  
    そのノードの活性化関数，そのノードのバイアス，そのノードに来ているシナプス荷重，そのノードに来ているシナプスの元ノード  
    がわかっていなければなりません  
    つまりあるノードの出力を知りたければその前のノードの出力を知る必要があり，その前の，その前の．．．となると  
    最終的に入力層のノードの出力を知る必要があるのですが，これはどこに保存されているのでしょう？

    - 多分保存されていないだろうし保存されていたとしてその複数の値をどのようにして扱えばいいのかわかりません  
    とりあえず今回はすべての入力が1だと過程して再帰的関数を実装していきます

    - 以下プログラム
    ```python
    def calculateOutput(connG, nodeG, focusID):
    _listo = []
    _listw = []
    _lista = []
    _val = 0

    # focusID を使用してfocusが目的地のシナプスを探す
    _indexc = np.where(connG[2, :] == focusID)
    _datac = connG[:, _indexc]

    # focusIDを使用してfocusを探す
    _indexn = np.where(nodeG[0, :] == focusID)
    _datan = nodeG[:, _indexn]

    # 入力層だったら出力は
    if(_datan[1, 0] == 0):
        output = 1 # 本来ここには入力ノードの出力が出ていないといけない
        return output

    # 隠れ層だったら
    # 目的地がfocusのシナプスすべてに対して
    for _i in range(_indexc.shape[0]):
        # 配列取得
        _datai = _datac[:, _i]

        # 出発地のノードを探す
        newfocusID = _datac[1]
        _val = calculateOutput(connG, nodeG, newfocusID)

        # ノードの出力とシナプス荷重の格納
        _listo.append(_val)
        _listw.append(_datai[3])
        _lista.append(_datai[4])

    output = 0
    for _i in range(len(_val)):
        if _lista[_i] == 1:
        output = output + _val[_i] * _listw[_i]

    return output
    ```

- バイアスについて
    - 今気づいたんだけどたぶんこのネットワークにバイアスは存在していません（これ普通に嘘かもしれない）  
    個体の情報を保存するところにコネクションデータとノードデータがあるんだけどノードデータの中にバイアスのデータを格納するようなところがない

## 2023/10/10
- 動かしてみたら動きませんでした
    - 直しました  
    ```python
    class WannInd(Ind):
    #省略
    # -- 'Single Weight Network' topological mutation ------------------------ -- #

    def topoMutate(self, p, innov,gen):
        #省略
        # Mutate Activation
        elif choice == 4:
        start = 1+self.nInput + self.nOutput
        end = nodeG.shape[1]           
        if start != end:
            mutNode = np.random.randint(start,end)
            branchData , act_change_alg = masudaSetup()

           #省略
            
            elif branchData['act_change_alg'] == 3:   #入力に対する出力の差
            sys.setrecursionlimit(10000)            #再帰関数のネストの深さを変更
            table1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            table2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # ランダムなノードの選択
            focusID = nodeG[0, mutNode]
            # ノードの出力を計算
            for _i in range (10) :
                table1[_i] = calculateOutput(connG, nodeG, focusID)
            
            #print(type(nodeG[2,mutNode])) #numpy.float64
            _ = random.choices(table2, weights = table1)[0]
            print(type(_))
            nodeG[2,mutNode] = _
            
            else:
            nodeG[2,mutNode] = nodeG[2,mutNode]

        child = WannInd(connG, nodeG)
        child.birth = gen

        return child, innov

    return data, table

    def calculateOutput(connG, nodeG, focusID):
    _listo = []
    _listw = []
    _lista = []
    _val = 0

    # focusID を使用してfocusが目的地のシナプスを探す
    _indexc = np.where(connG[2, :] == focusID)
    _datac = connG[:, _indexc]

    # focusIDを使用してfocusを探す
    _indexn = np.where(nodeG[0, :] == focusID)
    _datan = nodeG[:, _indexn]

    # 入力層だったら出力は
    if(_datan[1, 0] == 0):
        output = 1 # 本来ここには入力ノードの出力が出ていないといけない
        return output

    # 隠れ層だったら
    # 目的地がfocusのシナプスすべてに対して
    for _i in range(len(_indexc[0])):
        # 配列取得
        _datai = _datac[:, _i]

        # 出発地のノードを探す
        newfocusID = _datac[1]
        _val = calculateOutput(connG, nodeG, newfocusID)

        # ノードの出力とシナプス荷重の格納
        _listo.append(_val)         #ノード出力
        _listw.append(_datai[3])    #重み
        _lista.append(_datai[4])    #有効化してあるか

    output = 0
    for _i in range(len(_listo)):
        if _lista[_i] == 1:
        output = output + _listo[_i] * _listw[_i]

    return output
    ```
    
- 新しい手法
    - 活性化関数の変更において，変更してみたら評価があがった関数の組を保存し，将来ではその組が選ばれやすいようにするという手法  
    おもしろそう

## 2023/10/11
- 昨日は疲労のあまりそのままgitをpushしてしまいましたがコメントにある通りたまに動かなくなります
    - 200回中199回とかいてあるのは感覚で  
    個体トポロジーの変更は第2世代目の個体を生成するときに初めて行われるのですが  
    4世代目くらいまでうまく動いているときもあれば2世代目も生成できずエラーになるときもあります

    - print(_datac) で原因を見てみようと思いますが  
    ```python
    [[[120.]]

    [[  4.]]

    [[ 39.]]

    [[  1.]]

    [[  1.]]]
    [[[120.   444.  ]]

    [[  4.     6.  ]]

    [[ 39.    39.  ]]

    [[  1.    -1.27]]

    [[  1.     1.  ]]]
    [[[120.   444.  ]]

    [[  4.     6.  ]]

    [[ 39.    39.  ]]

    [[  1.    -1.27]]

    [[  1.     1.  ]]]
    
    File "/home/masuda/Documents/WANN/WANN/neat_src/wann_ind.py", line 246, in calculateOutput
    _datai = _datac[:, _i]
    IndexError: index 1 is out of bounds for axis 1 with size 1
    ```

    - 見るに要素数が1の列ベクトル的な配列に対しては正しい処理が行われているが，エラーが起こるときはなぜか2つの情報が入っており，サイズ1のところに要素数2のデータが入るわけないだろって怒られてる感じです  
    マルチコアで同時に書き込んでしまっているのかと思ったが2コア（masterとslave）でうごかしても要素数が増えるのでこれは原因ではないと思う

    - インデックスとかいう概念は理解できるんですけどそれを日常に取り込むのは気持ち悪いのでc言語ライクな実装にしてしまいました  
    ```python
    # 目的地がfocusIDのシナプスを抽出
    _datac = []
    for _i in range(connG.shape[1]):
        if connG[2, _i] == focusID:
        row = connG[:, _i]
        _datac.append(row)
    _datac = np.array(_datac)
    print(_datac)

    # focusIDを使用してfocusノードを探す
    _datan = []
    for _i in range(nodeG.shape[1]):
        if nodeG[0, _i] == focusID:
        row = nodeG[:, _i]
        _datan.append(row)
    _datan = np.array(_datan)
    print(_datan)
    ```
## 2023/10/13
- 進捗発表
    - また式が間違っています！
        - なにをしてんの  
        - 誤 $$ (f(\sum_{k=1}^{n}(n) - b) - g(\sum_{k=1}^{n}(n) - b))^2 $$  
        - 正 $$ d(f_{1}, f_{2}) = (f_{1}(\sum_{k=1}^{n}i_{n}) - f_{2}(\sum_{k=1}^{n}i_{n}))^2 $$  
        - $ d_{(f_{1}, _{2})} $ は $ f_{1} $ と $ f_{2} $ の距離(Distance)， $ f_{1} $ は現在の活性化関数 $ f_{2} $ はすべての活性化関数，1や2は活性化関数ID  
        $ \sum_{k=1}^{n}(i_{n}) $ はノードの出力，$ n $ はノードに何本の接続があるか，$ i_n $ は $ i $ 本目の入力値  
        $ b $ はバイアスのつもりで記載していたが，ノード情報にバイアスがないため削除

        - あとルーレット選択をするならQで終わりにしよう  
        - 修正前
        $$ P_{n} = \dfrac {Q_{n}}{\sum_{i} (Q_{i})} $$
        $$ Q_{i} = \begin{cases} \dfrac{1}{f_{x}} \qquad (x \neq t) \\ 0 \qquad (x = t) \end{cases} $$
        - 修正後
        $$ P_{i} =  \begin{cases} \dfrac{1}{d(f_{s}, f_{i}) + \varepsilon } \qquad (i \neq s) \\ 0 \qquad (i = s) \end{cases} $$
        - $ \varepsilon $ はとても小さい数，これは0や0に近い数値の逆数はエラーやとても大きい数になってしまうため  

        - 新しい式一覧  
        $$ d(f_{a}, f_{b}) = \int^{r}_{-r} (f_{a}(x) - f_{b}(x))^{2} $$
        $$ d(f_{a}, f_{b}) = (f_{a}(\sum_{k=1}^{n}i_{k}) - f_{b}(\sum_{k=1}^{n}i_{k}))^2 $$  
        $$ P_{i} =  \begin{cases} \dfrac{1}{d(f_{s}, f_{i}) + \varepsilon } \qquad (i \neq s) \\ 0 \qquad (i = s) \end{cases} $$
        | 変数                  | 意味                                       | 例                                 |
        | ----                  | ----                                       | ----                               |
        | $P_{i}$               | $s$から$i$へ活性化関数IDが変更される見込み | 0より大きい小数，                  |
        | $d(f_{s}, f_{i})$     | 活性化関数が$s$と$i$の距離                 | 0以上の小数(0なら同じ関数)         |
        | $f_{i}$               | IDが$i$の活性化関数                        | $tanh$とか$sigmoid$とか            |
        | $i$                   | 活性化関数ID                               | 1から10の整数                      |
        | $s$                   | 現在の活性化関数ID                         | 1から10の整数                      |
        | $\sum_{k=1}^{n}i_{k}$ | ノードに入力される値                       | 小数，各接続の合計                 |
        | $n$                   | ノードに接続されている入力シナプスの数     | 1以上の整数(0なら考慮する必要なし) |
        | $i_{k}$               | $k$本目の接続の入力値                      | 小数                               |
        | $\varepsilon$            | 逆数の大きさを抑えるための小さい値         | 0.1(自分で決める)                  |
        | $r$                   | 関数の考慮範囲                             | 0より大きい小数                    |

        - 実装は月曜日行おうと思います
    
    - 進捗報告をアップロード
        - しようと思います
    
## 2023/10/16
- おはようございます
    - あしたこそ朝マックが売っている時間に津田沼にいたい

- 二条誤差をプログラム実行時に計算できるようにしたい
    - 今まではWANNとは独立しているactfun.pyに計算をさせてその計算結果を手入力で配列に代入していたが，分解能や考慮する範囲やイプシロンを毎回指定したいのでビルトインした形式にしたい．
    - まずハイパラをmainからとってくる
    - 実行ファイルwann_train.pyにて以下を追加
    ```python
    parser.add_argument('-m', '--masuda', type=str,\
     help='卒業研究のために追加', default='p/masuda/branch.json')
    ```

- 修正した式への意見
    - 修正した式を教授にみせたところ色々意見をもらいました
    - $i$ は適切でない
        - どのようにして$i$を決めるのか，またそんな手法で関数が似ているかわかるのか
        - まず，入力$i$は一つのシナプスの入力ではなく，全部のシナプスの入力情報を持ったベクトルとして扱うのが普通らしい
        - またいつの入力を考慮すべきかは，タスクが始まってから終わるまでのうち，10フレームをランダムに抜き出した平均を取ろうと思っている．
        - よって新しい式一覧
            - 二条誤差の距離
            $$ d(f_{a}, f_{b}) = \int^{r}_{-r} (f_{a}(x) - f_{b}(x))^{2} $$
            - 入力に対する出力の距離
            $$ d(f_{a}, f_{b}) = (f_{a}(j) - f_{b}(j))^2 $$  
            $$ j =  (\dfrac{1}{n})\sum_{k=1}^{n}(input(random))$$
            - 距離から選ばれやすさを求める式
            $$ P_{i} =  \begin{cases} \dfrac{1}{d(f_{s}, f_{i}) + \varepsilon } \qquad (i \neq s) \\ 0 \qquad (i = s) \end{cases} $$
            | 変数                  | 意味                                       | 例                                 |
            | ----                  | ----                                       | ----                               |
            | $P_{i}$               | $s$から$i$へ活性化関数IDが変更される見込み | 0より大きい小数，                  |
            | $d(f_{s}, f_{i})$     | 活性化関数が$s$と$i$の距離                 | 0以上の小数(0なら同じ関数)         |
            | $f_{i}$               | IDが$i$の活性化関数                        | $tanh$とか$sigmoid$とか            |
            | $i$                   | 活性化関数ID                               | 1から10の整数                      |
            | $s$                   | 現在の活性化関数ID                         | 1から10の整数                      |
            | $f_{s}(j)$            | 入力$j$に対する出力                        | 小数                               |
            | $j$                   | 出力の距離を求めるために用いる入力         | 小数                               |
            | $input(random)$       | 複数回ある入力のうちのひとつ               | 1回目の入力，47回目の入力          |
            | $\varepsilon$            | 逆数の大きさを抑えるための小さい値         | 0.1(自分で決める)                  |
            | $r$                   | 関数の考慮範囲                             | 0より大きい小数                    |

            - 入力$i$は何百フレームとある動作の何百回の入力から出される．具体的にはランダムなフレーム目の入力をn回とってきて，合計して，nで割る．
            - input(random)は入力をランダムにとってくるが，これはノードに接続されているシナプスの信号の合計になる
        
        - あとイプシロンは$\varepsilon$ではなく$\varepsilon$を使うこと

## 2023/10/17
- 昨日の続き
    - まずグローバル変数としてmyHypをとりました  
    次に読み込んだパラメータ(resolution, calc_range, padding)から関数同士の誤差を計算します  
    ```python
    print('Loading myHyp...')
    global myHyp
    myHyp = loadMyHyp(args.masuda)
    print('Done.')
    ```

    ```python
    def loadMyHyp(MyHypPass):
    with open(MyHypPass, 'r') as json_file:
        MyHyp = json.load(json_file)
    
    resolution = MyHyp['resolution']
    calc_range = MyHyp['calc_range']
    padding = MyHyp['padding']

    _ = []
    _.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    MyHyp['activate_ID'] = _

    _ = [[0] * 10 for i in range(10)]

    for _i in range(len(_)):
        for _j in range(len(_[0])):
            _d = myDistance(resolution, calc_range, padding, _i, _j)
            _[_i][_j] = _d
    MyHyp['activate_table'] = _

    return MyHyp
    ```
    - myDistanceの処理は今まで使っていたactfun.pyからほぼほぼ流用
    - で活性化関数を変更してる関数とかクラスとかにmyHypを受け渡すことで処理を実現しました  
    変更箇所を知りたければmyHypとmyhypで検索してください

- スコアの保存について
    - 正直なんにもわかってない  
    - いろいろ出力されてるけどどれがどの個体の適応度なのかの説明も見当たらない

- 手法を変えて得られた結果
    - 既存手法 50世代目 swingup Elite449.79  
    提案手法 50世代目 swingup Elite476.59

    - Elite Fit: 共有重みを用いたその世代の中で最も適応度の高い個体の適応度  
    Best Fit: 今までのすべての世代の中で最も適応度の高い個体の適応度  
    Peak Fit: Best Fitの重みをチューニングした場合の適応度

- タスクを解いている最中のノードの出力がほしい
    - 入力層のノード出力がわかればそれでいいんだけど中間層の出力がわかると実装が楽
    - でどうやって値をとってくるのか
        - タスク（二足歩行とかポールのバランス取りとか）を実行するときに並列化をMPIによって行っているみたいです  
        wann_train.pyにて  
        ```python
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        ```

        - とありここでcommとrankが定義されている  
        comm.send(---) は各プロセッサにデータを与えているらしい  
        comm.Recv(---) は各プロセッサにお願いしてた計算結果を返してくれるらしい

        - 計算はdomain/bipedal_wolker.pyで行っているらしい[要出典]  
        追記 行っていないです，domain/config.pyで行っている[要出典]  
        わっかんね pushして帰ります

## 2023/10/20

- 昨日の続き
    - MPI.COMM.WORLDについての理解
        - https://mpi4py.readthedocs.io/en/stable/tutorial.html
        - https://keichi.dev/post/mpi4py/
    
    - 大文字と小文字について
        - WANNプログラムには以下のような記述がある  
        ```python
        comm.send(n_wVec, dest=(iWork)+1, tag=1)
        comm.Send(  wVec, dest=(iWork)+1, tag=2)
        comm.send(n_aVec, dest=(iWork)+1, tag=3)
        comm.Send(  aVec, dest=(iWork)+1, tag=4)
        ...
        comm.Recv(workResult, source=iWork)
        ```

        - comm.send と comm.Send は動作が異なる  
        comm.send は通常のpythonオブジェクトに対して記述される  
        comm.Send はメモリバッファを通信するために記述される  
        らしい

    - 文法の解説
        - このプログラムはデータを送信するために記述されていて，  
        送信する内容は n_wVec ，送信先のCPUコアIDは (iWork)+1 ，受信側との合言葉は 1 といった具合  
        ```python
        comm.send(n_wVec, dest=(iWork)+1, tag=1)
        ```
        - このプログラムはデータを受信するために記述されていて，  
        受信する内容は workResult(という名の変数に格納される) ，受信元のCPUコアIDは iWork  
        通常tagを指定するはずだが，chatGPTに聞いたら暗黙的にtag=0となるらしい  
        追記 なんか配列の長さが6あるので，すべてのtagの情報を受け取っているみたい
        ```python
        comm.Recv(workResult, source=iWork)
        ```
        追記 これ全然ウソついてますからこれから書くことが本当です
    
    - 本当に理解した（ほんと）
        - まず昨日から睨んでいたMPIの並列化についてだんだんわかってきた  
        でプログラムの流れを説明すると，  
        まず例によって wann_train.py からプログラムが実行され，並列化しコアID=0はマスター，ID!=0はスレーブとして働くことになる  
        で，マスターが  
        ```python
        # ばっさり省略
        for iBatch in range(nBatch): # Send one batch of individuals
            for iWork in range(nSlave): # (one to each worker if there)
            if i < nJobs:
                wVec   = pop[i].wMat.flatten()
                n_wVec = np.shape(wVec)[0]
                aVec   = pop[i].aVec.flatten()
                n_aVec = np.shape(aVec)[0]

                comm.send(n_wVec, dest=(iWork)+1, tag=1)
                comm.Send(  wVec, dest=(iWork)+1, tag=2)
                comm.send(n_aVec, dest=(iWork)+1, tag=3)
                comm.Send(  aVec, dest=(iWork)+1, tag=4)
                if sameSeedForEachIndividual is False:
                comm.send(seed.item(i), dest=(iWork)+1, tag=5)
                else:
                comm.send(  seed, dest=(iWork)+1, tag=5) 
            else: # message size of 0 is signal to shutdown workers
                n_wVec = 0
                comm.send(n_wVec,  dest=(iWork)+1)
            i = i+1 
        # ばっさり省略
        ```
        によって各スレーブコアに対して5つの情報を与えている  
        情報は 重み行列のサイズ，重み行列，活性化関数行列のサイズ，活性化関数行列，シート値 になる

        - で，このデータがどこで処理されるのかと言うと wann_train.py のslave関数から  
        ```python
        while True:
            n_wVec = comm.recv(source=0,  tag=1)# how long is the array that's coming?
            if n_wVec > 0:
            wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
            comm.Recv(wVec, source=0,  tag=2) # recieve weights

            n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?
            aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
            comm.Recv(aVec, source=0,  tag=4) # recieve it
            seed = comm.recv(source=0, tag=5) # random seed as int

            result = task.getFitness(wVec,aVec,hyp,seed=seed) # process it
            comm.Send(result, dest=0)            # send it back
        ```
        となる，ここでマスターが送った情報にはtag番号が振られており，この番号はマスターとスレーブで対応している必要がある  
        で，これらのネットワーク情報を使って， domain/wann_task_gym.py の getFitness 関数を呼び出し，この中でマスターが供給した情報のネットワークがどれくらいの適応度を持っているかを返り値として返す

        - この返り値をスレーブコアは  
        ```python
        comm.Send(result, dest=0)            # send it back
        ```
        にてマスターコアに送信している（マスターコアのIDは0，またこのときtagをつけていないのがむかつく）  
        マスター関数にて
        ```python
        for iWork in range(1,nSlave+1):
            if i < nJobs:
                workResult = np.empty(hyp['alg_nVals'], dtype='d')
                comm.Recv(workResult, source=iWork)
                reward[i,:] = workResult
            i+=1
        return reward
        ```
        で情報を受け取っている  
        この値をrewaedに格納することで，個体の評価を保存することができている

    - じゃあ getFitness が気になってくるよねって話
        - domain/task_gym.py を呼び出してるみたいで，見てみると  
        ```python
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)  
        action = selectAct(annOut,self.actSelect)    
        ```
        とあり，これまんま入力と出力じゃね！？！？！？！？！？！？！？  
        しかし今日は帰ります，Greed is Good をプレイするので．  
        あと既存手法で一晩まわしてみようかな

## 2023/10/24
- 今日すずし
    - ついに夏終わったか

- 前回の続き
    - まずこの通信がどの頻度で行われているかを確認する必要がある  
    学習するには，まず世代を回す必要がある（1世代に1回個体の更新が行われる）  
    1世代の評価には複数回の入力がある（300フレーム生きていたら300入力）
    - このマスターとスレーブの情報の受け渡しがどの粒度で行われているか
    ```python
    def master(): 
    for gen in range(hyp['maxGen']):        
        reward = batchMpiEval(pop)  # 生成した個体の評価
    ```
    より1世代に1回batchMpiEvalは呼び出されている
    ```python
    def batchMpiEval(pop, sameSeedForEachIndividual=True):
        nSlave = nWorker-1
        nJobs = len(pop)
        nBatch= math.ceil(nJobs/nSlave)
        
        for iBatch in range(nBatch): # Send one batch of individuals
            for iWork in range(nSlave): # (one to each worker if there)
                comm.send(n_wVec, dest=(iWork)+1, tag=1)
                comm.Send(  wVec, dest=(iWork)+1, tag=2)
                comm.send(n_aVec, dest=(iWork)+1, tag=3)
                comm.Send(  aVec, dest=(iWork)+1, tag=4)
                if sameSeedForEachIndividual is False:
                comm.send(seed.item(i), dest=(iWork)+1, tag=5)
                else:
                comm.send(  seed, dest=(iWork)+1, tag=5)        
        
            for iWork in range(nSlave):
                if i < nJobs:
                    comm.Recv(workResult, source=(iWork)+1)
    ```
    for iWork in range(nSlave):  
        source=(iWork)+1  
    と  
    for iWork in range(1, nSlave+1):  
        source=iWork  
    が混在しているので，注意すること
    - より，1世代の間では nBatch x nSlave 回のデータやりとりをしているみたいで，各スレーブコアに対してデータを送ったあとにdef slave()で計算，またマスターコアに計算結果を送って保存する  
    nBatch = nWorker - 1 = 12 - 1 = 11 コア数-1つまりスレーブコアの数分だけ  
    nJobs = 480(今回に限って) 個体の数つまり480人が転ぶまで計算している  
    nBatch = 480 / 11 = 44 推測だけどスレーブコア1コアの働き分？

    - つまり，スレーブコア1つずつに44人分の計算をさせている  
    それで44人を11コアで世代全員分の480人分の計算をする  
    ID1のコアには0から43までの計算，ID2のコアには44から87までの計算，，ID11のコアには435から479までの計算  

    - master が slave に情報を渡して， slave は getFitness を呼び出して， getFitness は testInd を呼び出し，その中で
    ```python
    for tStep in range(self.maxEpisodeLength): 
      annOut = act(wVec, aVec, self.nInput, self.nOutput, state) 
      print(state)
    ```
    とすると大きさ24の配列がプリントされて，明らかにエージェントの状態を追跡できている
    ```
    [ 1.15e+00  1.25e-01 -2.10e-02 -5.97e-02 -3.02e-01 -8.01e-01 -6.25e-01 -3.93e-04  1.00e+00 -8.35e-01 -5.13e-06 -6.21e-01 -5.44e-06  0.00e+00  3.30e-01
    3.34e-01  3.46e-01  3.67e-01  4.00e-01  4.52e-01  5.32e-01  6.64e-01  9.12e-01  1.00e+00]
    [ 1.21e+00  1.19e-01 -2.69e-02 -5.86e-02 -3.61e-01 -7.49e-01 -6.25e-01 -4.02e-04  1.00e+00 -8.35e-01 -5.36e-06 -6.18e-01 -7.47e-06  0.00e+00  3.28e-01
    3.31e-01  3.43e-01  3.64e-01  3.97e-01  4.48e-01  5.27e-01  6.59e-01  9.04e-01  1.00e+00]
    [ 1.26e+00  1.12e-01 -3.38e-02 -5.65e-02 -4.16e-01 -6.86e-01 -6.25e-01 -5.71e-04  1.00e+00 -8.35e-01 -7.75e-06 -6.16e-01 -1.28e-05  0.00e+00  3.25e-01
    3.29e-01  3.40e-01  3.61e-01  3.94e-01  4.44e-01  5.23e-01  6.53e-01  8.97e-01  1.00e+00]
    ```
    あとはこれを保存して，考慮して，変更確率を計算する

    - 書き込み場所間違えました  
    いややっぱあってるかも，多分あってる  
    今日はここまで，あしたは午前から来る

## 2023/10/26
- あしたは午前から来れるわけもなく，，来たのは2日後の午後でした．

- 昨日の続き
    - 入力ノードの出力はprintできるようになったので，これを個体数分用意する

    - 未来の自分のためにステップをまとめます  
    1. testInd内で適応度を求める際に当然実際にエージェントを動かしているので，毎ステップの入力ノードの情報を記録します
    2. 各個体（生まれ，歩き，最後に転ぶまで）の入力ノードの集まりをハイパーパラメータmini_batch_sizeの数だけランダムに取ります，このときの「各個体の入力ノードの集まり」には，-2.0, -1.0, -0,5 +0.5, +1.0, +2.0の**いずれか1つの**共有重みを使ったときのすべての入力が入っています．
    3. 取った値の平均を求めます，これがbatched_stateになります．BipedalWalker-v2の場合batched_stateはfloatが24個入った配列になります．
    4. 1から3の処理はすべてスレーブコアによって行われているため，まずlen(batched_state)の値，つまり入力ノードの数をマスターコアに送信します．これは，コア間の通信には要素の型と数を明示的に指定する必要があるためです．len(---)は常にintで1つなので問題なく送れます．
    5. マスターコアでは，スレーブコアから受信した入力ノードの数から，ノードの数分だけの要素数をもったnp.arrayを宣言します．このあとすぐにデータを詰め込むので，初期化しないかわりに動作の早いnp.emptyを使います．
    6. スレーブコアはbatched_stateをマスターコアに送信します．
    7. マスターコアはbatched_stateをスレーブコアから受信します．このとき配列はすでに作っているので，batched_stateがどのような大きさであっても通信は成功します．  

    結果として，6 x 24 (-2.0, -1.0, -0,5 +0.5, +1.0, +2.0) x (バッチ処理した入力ノード) を得ることができます
    testIndが吐くのが2次元配列なのでnp.arrayにできないみたいなことがわかったので今日は帰ります

## 2023/10/31
- 10月も終わるというのにおれの研究は．
    - とりあえず.xbbファイルを作成するスクリプトをかきました＾＾  
    実行は . ./xb.sh より

- 前回は配列に保存する際，二次元リストを用いてそのひとつひとつの要素に1次元リストを格納しようとしたらデカすぎますって怒られたところから
    - そこはうまくいって今はcomm.Sendで送る内容が4要素までにしてくれって怒られてるっぽいんだけどじゃあなんでfitnessは要素数6でいけてるんだよって感じです
    - うごきまーーーしたーーー！！！

## 2023/11/2
- 今月食費けずりチャンレンジしてみます，初日の今日はさっそくランチパックを大量に購入しｵﾜ．
- 入力ノード情報を活性化関数変更モジュールまで持っていく
    - 過去に活性化関数変更モジュールに接続情報をもっていったのと同じように変数を渡してやる  
    これ普通に過去に受け渡しはやってなくて泣いた
    - 次世代の個体をする前に適応度順に個体を並べ替えているので，入力ノード情報も同じ組みあわせで入れ替える必要がある．
    ```python
    # Sort by rank
    ranklist = []
    for _i in range(len(pop)):
        _r = vars(pop[_i])['rank']
        ranklist.append(_r)
    staterank = list(zip(state, ranklist))

    pop.sort(key=lambda x: x.rank)
    staterank.sort(key=lambda x: x[1])

    state, ranklist = zip(*staterank)
    ```
    - まず個体の適応度情報を入力ノード情報と一緒にする(zip)，で適応度を基準に並べ替えた後にstateだけ切り離してる

- とりあえずcalculateOutputまではstateを持っていってエラーも出ていない
- バスで変えるので少し早めに失礼します

## 2023/11/3
- 昨日の続き 入力ノード情報の利用
    - 使いたい関数にまでは持ってきたので，あとはコーディングをまちがえないようにしたい  
    未来の自分が理解しやすいように疑似コードを用いて記しておく
    ```
    for n in 活性化関数 {
        for m in 共有重み {
            for o in ミニバッチサイズ {
                table[n] += nodeOut(活性化関数を変更したいノードID, n, m, list[活性化関数を変更したい個体][m][o])
            }
        }
    }

    function nodeOut(該当ノード, 該当関数, 共有重み, 入力ノードリスト){
        for i in 目的地が該当ノードの接続 {
            if iの出発地が入力層 {
                前出力 += 入力ノードリスト[iの出発地]
            }
            if iの出発地が隠れ層 {
                前出力 += nodeOut(iの出発地のノードID, 活性化関数ID, 共有重み, 入力ノードリスト)
            }
        }
        入力 = 前出力 * 共有重み
        出力 = 該当関数(入力)
        return 出力
    }
    ```

- 定期発表
    - またまた式が間違っていますね
        - なにを四天王バカカスが代
        - 間違った式  
        $d(f_{a}, f_{b}) = (\sum_{m}(f_{a}(in_{m})) - \sum_{m}(f_{b}(in_{m})))^2$

        - 訂正した式  
        $d(f_{a}, f_{b}) = \sum_{m}(f_{a}(in_{m}) - f_{b}(in_{m}))^2$

    - 卒業研究をどこまで行うのか
        - シナプス荷重を細かく振り分けるのはどう考えてもノンワースなので他の方法を考えて入るんだけおd全くアイデアが出ません
    - 関数同士の「距離」というのは本当に正しい表現なのか  
    3つのもの$x, y, z$が存在し，$d$を考えるときに，どんな$x, y, z$に対しても
        - 非負性
            - $d(x, y) >= 0$
        - 同一律
            - $d(x, y) = 0(x=y)$
        - 対象律
            - $d(x, y) = d(y, x)$
        - 三角不等式
            - $d(x, y) <= d(x, z) + d(z, y)$
    - が成り立てばdは距離関数としてみて問題無いだろうということ

    - $d(f_{a}, f_{b}) = \sum_{m}(f_{a}(in_{m}) - f_{b}(in_{m}))^2$ は距離関数として成立しているか
        - 非負性
            - $f_{a}(in_{m})$ と $f_{b}(in_{m})$ は両者とも実数であるため，$(f_{a}(in_{m}) - f_{b}(in_{m}))$ もまた実数と言える  
            実数$x$において$x^2$は常に0以上の実数より，$(f_{a}(in_{m}) - f_{b}(in_{m}))^2$ もまた0以上の実数と言える  
            0以上の$x$において$\sum(x)$は常に0以上の実数になるので $\sum_{m}(f_{a}(in_{m}) - f_{b}(in_{m}))^2$ もまた0以上の実数であるとわかり，$d$は非負性を満たしている
        
        - 同一律(xとy)が等しい場合のみ，距離は0になる性質(11/7修正)
            - $d(x, x)$  
            $= d(f_{a}, f_{a})$  
            $= (\sum_{m}(f_{a}(in_{m})) - \sum_{m}(f_{a}(in_{m})))^2$  
            $= (\sum_{m}(0)^2)$  
            $= (\sum_{m}(0))$  
            $= 0$  
            ただし，各ミニバッチにて $\sum_{m}(f_{a}(in_{m})) = \sum_{m}(f_{a}(in_{m}))$ の場合は距離が0になってしまうので同一律は満たしていない．  
            d(x, x) = 0 と d(x, y) = 0 if only x = y  
            は大きく意味が異なることに注意
        
        - 対象律
            - 実数$x$において  
            $(x)^2 = (-x)^2$ より，$x = f_{a}(in_{m}) - f_{b}(in_{m})$ とすると $-x = -f_{a}(in_{m}) + f_{b}(in_{m})$ より，  
            $(f_{a}(in_{m}) - f_{b}(in_{m}))^2 = (f_{b}(in_{m}) - f_{a}(in_{m}))^2$ が成り立つ  
            これより$d$は対象律を満たしている
        
        - 三角不等式
            - https://iwai-math-blog.com/index.php/distance-function/  
            を参考に証明できました，詳しくはgithub無いmasudaでまとめてありますのでそちらを見てください

## 2023/11/7

- よく考えたら同一律満たしてませんでした
    - 同一律を満たしていない距離関数をどう扱うか
        - 同一律を満たしていないことは想定の範疇というか，もとからある入力に対する出力さえ近ければそれは近いと見なす，という意見だったので，同一律をみたいしていないことはデメリットにはならない   
        同一律を満たさないが， $d(x, x) = 0$ を満たす関数は「擬距離」と呼ばれる  
        これからは擬距離呼びで表記する
    
- $ \varepsilon $ の値を進捗状況にあわせて変化させる
    - 活性化関数を変えることで，それまでの良い結果を反転させてしまうことを憂いていましたが，そもそも良い結果じゃないのにその付近をウロウロするのは頭が悪いと言わざるを得ません  
    それまでの良い結果を反転させてしまう可能性があることは，それまでの悪い結果を反転してくれる可能性があることにも注目しましょう  
    よって，$ \varepsilon $ の値を  
    最初は大きく，つまり変更確率は関数同士の距離にあまり影響しないように  
    最後は小さく，つまり変更確率は関数同士の距離に大きく影響するようにすることがさらなる精度の向上につながると確信します

    - $ \varepsilon $ の決定方法
        - まず，$ \varepsilon_{0} $ と $ k $ をハイパーパラメータとして定義しておく  
        以下 $ \varepsilon_{n} $ は以下の式に従い変動する  
        $ \varepsilon_{n} = k * \varepsilon_{n-1} $
        
        - kを1よりほんの少し小さい数にすることで，世代数が大きくなるにつれて $ \varepsilon_{n} $ の値は小さくなり，期待した動作を保証する  
        例として，  
        $ \varepsilon_{0} = 1.0, k = 0.99 $ とすると， $ \varepsilon_{100} =  0.36, \varepsilon_{1000} = 0.00004, \varepsilon_{2048} =  0.000000001$  
        $ \varepsilon_{0} = 1.0, k = 0.999 $ とすると， $ \varepsilon_{100} =  0.90, \varepsilon_{1000} = 0.36, \varepsilon_{2048} =  0.129$  
        といった具合になる
    
- 実装していきます
    - ハイパーパラメータ
        - こんな感じで  
        ```json
        {       
            ...
            "ratio" : 0.99,
            "epsilon": 0.1,
            ...
        }
        ```
        - ratioに $ k $ を， epsilonに $ \varepsilon $ を入れます
    
    - 計算に組み込む
        - まず $ \varepsilon $ を抜いた計算テーブルをmyHypに用意する必要がある

## 2023/11/10
- 実行結果
    - 3日間動かしてみたんですけど，二条差で実行したときよりも，入力に対する出力の差のほうがスコアが10倍とかになってしまったので，これはなにかおかしいということでもう一度実行し直しています．
    - 結果は，二条差が500世代目で1.89に対して入力に対する出力の差では15とかです．

## 2023/11/16
- 前回の訂正
    - 実行していくと120世代目で急にスコアが改善して，提案手法と大差なくなりました  
    結果的には微プラスって感じです

- お別れの時間です
    - 実装自体は終わったので，ここで知見を書き込むこともなくなるでしょう  
    いままでありがとうございました

## 2023/11/22
- お久しぶりですまたかきます

- 提案手法間違ってね
    - 既存手法との有意差を見つけられないのでこれはプログラムが間違っていて今動いているのはランダムに近いものなんじゃないかと推測します

- 確認
    - プログラムを一部切り抜いてデバッグしてみる
    - うまくいきました  
    ノードID0は入力層ではなくバイアスをかけるための専用ノードだったので，0から23の範囲を1から24に変更しました．