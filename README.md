## 1、使用脚本将Realman hdf5格式 转成 lerobot 数据集：（xyz+轴角）

### （1）单任务转换

**hdf5_dir** Realman数据集目录

**output_dir** 输出lerobot目录

**task_name** 任务名称

**fps 20** video帧数

```bash
python /home/jdtech/Documents/dataset_convert/mydataset2lerobot_2angle_one_task.py \
    --hdf5_dir ./lerobot_cup_test \
    --output_dir ./take_cup_test \
    --task_name take_cup \
    --fps 20
```

**转换前后格式对比：**

直接将 Realman 录制的数据集 转为 lerobot 数据集格式

1. **Realman：**

Group: action

Dataset: action/arm_left

Dataset: action/arm_right

Dataset: action/hand_left

Dataset: action/hand_right



Group: observations

Dataset: observations/arm_left

Dataset: observations/arm_right

Dataset: observations/hand_left

Dataset: observations/hand_right



Group: observations/images

Dataset: observations/images/cam_head



2. **lerobot：**

​	data

​	meta

​	video

### （2）多任务转换

**tasks_dir** 包含多个任务的数据集的总目录

**output_dir** 输出的lerobot目录

**fps** video帧数



**realman多任务hdf5保存示例**：程序会将dataset的子目录的名作为每个任务的 task description，如“take_cup"。



dataset

​	——take_cup

​		——episode1.hdf5

​		——episode2.hdf5

​		……

​	——catch_loopy

​		……

​	——



```bash
python mydataset2lerobot_2angle_all_task.py \
    --tasks_dir /home/jdtech/Documents/dataset_convert/dataset \
    --output_dir /home/jdtech/Documents/dataset_convert/dataset_lerobot  \
    --fps 20
```



## 2、在/meta/中手动添加在模型中需要使用的 modality.json文件（gr00t模型）

当模型需要使用双臂数据时：

**state**: 

​	arm_left:

​		joint [0,6] ; 

​		position[6,12]; xyz+轴角

​	arm_right:

​		joint [12,18] ; 

​		position[18,24]; xyz+轴角

​	hand_left:

​		angle(0-2000) [24,30]

​		pose(0-1000)[30,36]

​		force[36,42]

​	hand_left:

​		angle(0-2000) [42,48]

​		pose(0-1000)[48,54]

​		force[54,60]

**action:**

​	arm_left:

​		position [0,6] ; xyz+轴角

​	arm_right:

​		position [6,12] ;xyz +轴角

​	hand_left:

​		pose(0-1000)[12,18];

​	hand_right:

​		pose(0-1000)[18,24];


modality.json文件:
```json
{
    "state": {
        "arm_left": {
            "start": 6,
            "end": 12
        },
        "arm_right": {
            "start": 18,
            "end": 24
        },
        "hand_left": {
            "start": 30,
            "end": 36
        },
        "hand_right": {
            "start": 48,
            "end": 54
        }
    },
    "action": {
        "arm_left": {
            "start": 0,
            "end": 6
        },
        "arm_right": {
            "start": 6,
            "end": 12
        },
        "hand_left": {
            "start": 12,
            "end": 18
        },
        "hand_right": {
            "start": 18,
            "end": 24
        }
    },
    "video": {
        "cam_head_left": {
            "original_key": "observation.images.cam_head_left"
        },
        "cam_head_right": {
            "original_key": "observation.images.cam_head_right"
        }
    },
    "annotation": {
        "human.action.task_description": {
            "original_key": "task_index"
        }
    }
}