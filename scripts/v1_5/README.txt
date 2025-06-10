说明：数据集和模型的地址相应替换成你那边服务上的地址

***** pretrain *****  （4卡）

pretrain.sh中：

配置一：直接pretrain，不加<MLCoT>token （pretrain baseline）
             --use_MLCoT False   （--MLCoT_num 10 不用修改）
             --output_dir ./checkpoints/viscot-7b-224-pretrain-baseline
             --per_device_train_batch_size 256

配置二：pretrain中加<MLCoT>token （pretrain MLCoT-v1）
             --use_MLCoT True
             --output_dir ./viscot-7b-224-pretrain-MLCoT-v1
             --per_device_train_batch_size 256

说明：先跑完上面两个pretrain再跑下面的finetune，我这边已经训完了配置二，配置一还在训，你可以直接下载我服务器上训练完的模型，下载完之后你可以把路径名称改一下（把-new去掉）保持和下面一致
配置一地址：/hdd/wuwl/project/MLCoT/checkpoints/viscot-7b-224-pretrain-baseline-new/ （还在训练中，大概剩余两小时）
配置二地址：/hdd/wuwl/project/MLCoT/checkpoints/viscot-7b-224-pretrain-MLCoT-v1-new/

***** finetune *****（试试4卡能不能跑起来）

finetune.sh中：

配置一：在pretrain baseline基础上，不加<MLCoT>token，直接finetune （StdPT-StdFT）
             --use_MLCoT False
             --pretrain_mm_mlp_adapter ./checkpoints/viscot-7b-224-pretrain-baseline/   （注意路径和上面命名一致）
             --output_dir ./checkpoints/viscot-7b-224-StdPT-StdFT
             --per_device_train_batch_size 128

配置二：在pretrain MLCoT-v1基础上，加<MLCoT>token进行finetune （MLCoTPT-MLCoTFT）
             --use_MLCoT True
             --pretrain_mm_mlp_adapter ./checkpoints/viscot-7b-224-pretrain-MLCoT-v1/   （注意路径和上面命名一致）
             --output_dir ./checkpoints/viscot-7b-224-MLCoTPT-MLCoTFT-v1
             --per_device_train_batch_size 128

配置三：在pretrain baseline基础上，加<MLCoT>token进行finetune （StdPT-MLCoTFT）
             --use_MLCoT True
             --pretrain_mm_mlp_adapter ./checkpoints/viscot-7b-224-pretrain-baseline/   （注意路径和上面命名一致）
             --output_dir ./checkpoints/viscot-7b-224-StdPT-MLCoTFT-v1
             --per_device_train_batch_size 128

配置四：在pretrain MLCoT-v1基础上，不加<MLCoT>token，直接finetune （MLCoTPT-StdFT）
             --use_MLCoT False
             --pretrain_mm_mlp_adapter ./checkpoints/viscot-7b-224-pretrain-MLCoT-v1/   （注意路径和上面命名一致）
             --output_dir ./checkpoints/viscot-7b-224-StdPT-MLCoTPT-StdFT-v1
             --per_device_train_batch_size 128

说明：跑完这四组finetune之后在测试集上测试一下，然后可以试试不同的--MLCoT_num参数，仿照论文里，测试一下2，5，25，50
