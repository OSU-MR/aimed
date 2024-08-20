#healthy subjects idx range [1~21]
#patient subjects idx range [93~99]
#digital subjects idx range [-13~-24]

#pre_defined dataset index
dataset_idx = [
    {   #train: healthy, validate: healthy, test: patient
        'train': [10, 3, 4, 6, 11, 1],
        'val': [2, 5, 18, 19, 20, 21],
        'test': [99, 98, 97, 96, 93]
    },
    {   #train: healthy, validate: healthy, test: healthy
        'train': [10, 3, 4, 6, 11, 1],
        'val': [2, 5, 18],
        'test': [19, 20, 21]
    },
    {   #train: digital, validate: digital, test: digital
        'train': [-15, -16, -17, -20, -21, -22],
        'val': [-14, -23, -13],
        'test': [-18, -19, -24]
    }
]

#pre_defined dataset index for demo evaluation
dataset_idx_demo = [
    {   #test: digital
        'train': None,
        'val': None,
        'test': [10000],# 
        'demo_slice_idx' : 0
    },
    {   #test: healthy subject (3T)
        'test': [20000],
        'demo_slice_idx' : 0
    },
    {   #test: patient (1.5T)
        'train': None,
        'val': None,
        'test': [98],
        'demo_slice_idx' : 0
    },
    {   #test: patient (0.55T)
        'train': None,
        'val': None,
        'test': [1983],
        'demo_slice_idx' : 0
    },

]