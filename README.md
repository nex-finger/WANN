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
        $$ P_{i} =  \begin{cases} \dfrac{1}{d(f_{s}, f_{i}) + \epsilon } \qquad (i \neq s) \\ 0 \qquad (i = s) \end{cases} $$
        - $ \epsilon $ はとても小さい数，これは0や0に近い数値の逆数はエラーやとても大きい数になってしまうため  

        - 新しい式一覧  
        $$ d(f_{a}, f_{b}) = \int^{r}_{-r} (f_{a}(x) - f_{b}(x))^{2} $$
        $$ d(f_{a}, f_{b}) = (f_{a}(\sum_{k=1}^{n}i_{k}) - f_{b}(\sum_{k=1}^{n}i_{k}))^2 $$  
        $$ P_{i} =  \begin{cases} \dfrac{1}{d(f_{s}, f_{i}) + \epsilon } \qquad (i \neq s) \\ 0 \qquad (i = s) \end{cases} $$
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
        | $\epsilon$            | 逆数の大きさを抑えるための小さい値         | 0.1(自分で決める)                  |
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
            $$ P_{i} =  \begin{cases} \dfrac{1}{d(f_{s}, f_{i}) + \epsilon } \qquad (i \neq s) \\ 0 \qquad (i = s) \end{cases} $$
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
            | $\epsilon$            | 逆数の大きさを抑えるための小さい値         | 0.1(自分で決める)                  |
            | $r$                   | 関数の考慮範囲                             | 0より大きい小数                    |

            - 入力$i$は何百フレームとある動作の何百回の入力から出される．具体的にはランダムなフレーム目の入力をn回とってきて，合計して，nで割る．
            - input(random)は入力をランダムにとってくるが，これはノードに接続されているシナプスの信号の合計になる
        
        - あとイプシロンは$\epsilon$ではなく$\varepsilon$を使うこと

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