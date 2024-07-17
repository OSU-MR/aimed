import os
import numpy as np
import nibabel as nib
import argparse



def get_datasets(patient_list, dataset_name, base_path, sub_path, view_list, suffix):
    if not patient_list:
        print(f"No patients in the list for {dataset_name}")
        return {}
    
    x_data = {}
    for patient_id in patient_list:
        for view_name in view_list:
            file_path = os.path.join(base_path, 'niidata' + suffix, f'patient{patient_id}_{view_name}.nii.gz')
            print(f"Loading {file_path}")

            try:
                data = nib.load(file_path).get_fdata()
                if data.shape[-2] == 256:
                    data = data[..., 32:-32, :]
                elif data.shape[-1] == 256:
                    data = data[..., :, 32:-32]
                else:
                    raise ValueError(f"Unexpected data shape: {data.shape}")

                if view_name in x_data:
                    x_data[view_name] = np.vstack((x_data[view_name], data))
                else:
                    x_data[view_name] = data
                
            except Exception as e:
                print(f"Error loading data for patient {patient_id}, view {view_name}: {e}")

    print_dataset_shapes(dataset_name, x_data)
    return x_data

def print_dataset_shapes(dataset_name, x_data):
    for view_name, data in x_data.items():
        num_patients = data.shape[0] // 8 if data.shape[0] // 8 > 0 else data.shape[0]
        print(f"{dataset_name}, {num_patients} patients, {view_name} shape: {data.shape}")



def load_data(args):
    base_path = os.getcwd() + '/'
    print(f"Base path: {base_path}")

    if args.dataset_shape not in ["256by192"]:
        raise ValueError(f"Unsupported dataset shape: {args.dataset_shape}")

    sub_path = base_path + 'niidata' + args.suffix if args.dataset_shape == "256by192" else ValueError(f"Unsupported dataset shape: {args.dataset_shape}")
    view_list = args.view_list

    data_train = get_datasets(args.patients_train, "training", base_path, sub_path, view_list, args.suffix)
    data_val = get_datasets(args.patients_val, "validation", base_path, sub_path, view_list, args.suffix)
    data_test = get_datasets(args.patients_test, "testing", base_path, sub_path, view_list, args.suffix)

    if 'SAX' in data_test:
        vol_shape = [data_test['SAX'].shape[-2], data_test['SAX'].shape[-1]]
        print("Testing volume shape:", vol_shape)

    return data_train, data_val, data_test, vol_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataloader options: 256by192')
    parser.add_argument('--dataset_shape', default="256by192", type=str, help='Shape of loaded dataset')
    parser.add_argument('--patients_train', nargs='+', default=[1,2,3,4,5,6], help='Patient numbers for training')
    parser.add_argument('--patients_val', nargs='+', default=[10,11], help='Patient numbers for testing')
    parser.add_argument('--patients_test', nargs='+', default=[98], help='Patient numbers for true testing')
    parser.add_argument('--view_list', nargs='+', default=['LAX','SAX','2CH'], help='Views for loading')
    parser.add_argument('--suffix', default='', help='Suffix for loading the dataset')
    args = parser.parse_args()

    load_data(args)

