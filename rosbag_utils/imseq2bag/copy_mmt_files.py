import os
import shutil

# Base Paths
SOURCE_BASE_PATH = "/mnt/usb-USB_3.0_Device_1_000000004858-0:1-part1/Unmanned_DB/4th"
TARGET_BASE_PATH = "/home/snu/DATA/4th-year-dynamic/003"

# Changeable Values
FOLDER = "1-02d"
START_SCENE_TIME = "151948_00"
END_SCENE_TIME = "152030_09"


def get_selected_file_path_list(modal_base_path, start_scene_time, end_scene_time):
    # Get Modal Data List
    modal_data_file_list = sorted(os.listdir(modal_base_path))

    # Find Start and End Index
    modal_start_file_idx = [idx for idx, s in enumerate(modal_data_file_list) if start_scene_time in s]
    assert len(modal_start_file_idx) == 1
    modal_start_file_idx = modal_start_file_idx[0]
    modal_end_file_idx = [idx for idx, s in enumerate(modal_data_file_list) if end_scene_time in s]
    assert len(modal_end_file_idx) == 1
    modal_end_file_idx = modal_end_file_idx[0]

    # Get Selected Files
    modal_sel_file_list = modal_data_file_list[modal_start_file_idx:modal_end_file_idx+1]

    # Append Modal Base Path to get Total Path
    modal_sel_file_path_list = \
        [os.path.join(modal_base_path, file_name) for file_name in modal_sel_file_list]

    return modal_sel_file_path_list


def gather_selected_modal_paths():
    # Get RGB Files
    RGB_SOURCE_BASE_PATH = os.path.join(SOURCE_BASE_PATH, FOLDER, "rgbdepth", "color")
    RGB_SEL_DIR_LIST = get_selected_file_path_list(
        modal_base_path=RGB_SOURCE_BASE_PATH,
        start_scene_time=START_SCENE_TIME, end_scene_time=END_SCENE_TIME
    )

    # Get Aligned Depth Files
    ADEPTH_SOURCE_BASE_PATH = os.path.join(SOURCE_BASE_PATH, FOLDER, "rgbdepth", "aligned_depth")
    ADEPTH_SEL_DIR_LIST = get_selected_file_path_list(
        modal_base_path=ADEPTH_SOURCE_BASE_PATH,
        start_scene_time=START_SCENE_TIME, end_scene_time=END_SCENE_TIME
    )

    # Get Infrared Files
    IR_SOURCE_BASE_PATH = os.path.join(SOURCE_BASE_PATH, FOLDER, "rgbdepth", "ir")
    IR_SEL_DIR_LIST = get_selected_file_path_list(
        modal_base_path=IR_SOURCE_BASE_PATH,
        start_scene_time=START_SCENE_TIME, end_scene_time=END_SCENE_TIME
    )

    # Get Thermal Files
    THERMAL_SOURCE_BASE_PATH = os.path.join(SOURCE_BASE_PATH, FOLDER, "thermal")
    THERMAL_SEL_DIR_LIST = get_selected_file_path_list(
        modal_base_path=THERMAL_SOURCE_BASE_PATH,
        start_scene_time=START_SCENE_TIME, end_scene_time=END_SCENE_TIME
    )

    # Get Nightvision Files
    NV_SOURCE_BASE_PATH = os.path.join(SOURCE_BASE_PATH, FOLDER, "nv1")
    NV_SEL_DIR_LIST = get_selected_file_path_list(
        modal_base_path=NV_SOURCE_BASE_PATH,
        start_scene_time=START_SCENE_TIME, end_scene_time=END_SCENE_TIME
    )

    # Get LiDAR Files
    LIDAR_SOURCE_BASE_PATH = os.path.join(SOURCE_BASE_PATH, FOLDER, "lidar")
    LIDAR_SEL_DIR_LIST = get_selected_file_path_list(
        modal_base_path=LIDAR_SOURCE_BASE_PATH,
        start_scene_time=START_SCENE_TIME, end_scene_time=END_SCENE_TIME
    )

    # Init Modal Dict
    return {
        "RGB": RGB_SEL_DIR_LIST, "aligned_depth": ADEPTH_SEL_DIR_LIST,
        "infrared": IR_SEL_DIR_LIST, "thermal": THERMAL_SEL_DIR_LIST,
        "nightvision": NV_SEL_DIR_LIST, "lidar": LIDAR_SEL_DIR_LIST
    }


if __name__ == "__main__":
    modals_selected_file_dict = gather_selected_modal_paths()
    modals_selected_file_dict["thermal"] = []

    # Check if Target Folder is Empty
    if len(os.listdir(TARGET_BASE_PATH)) == 0:
        for modal, modal_sel_file_list in modals_selected_file_dict.items():
            # Make Modal Folders
            modal_target_dir = os.path.join(TARGET_BASE_PATH, modal)
            if os.path.isdir(modal_target_dir) is False:
                os.mkdir(modal_target_dir)

            # Copy Files
            for idx, modal_file_path in enumerate(modal_sel_file_list):
                modal_file_name = modal_file_path.split("/")[-1]
                modal_target_file_path = os.path.join(modal_target_dir, modal_file_name)

                # Copy
                shutil.copyfile(modal_file_path, modal_target_file_path)

                # Process String
                copy_process_str = "Copying Modal [{}] - ({}/{})".format(
                    modal, idx+1, len(modal_sel_file_list)
                )
                print(copy_process_str)
