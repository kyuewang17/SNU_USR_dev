import os
import logging
import datetime
from zipfile import ZipFile


__BASE_PATH__ = "/media/kyle/DATA003/mmosr_RAL_db/MMOSDX_DB_RESULTS/iitp_2021_mot_results"


def set_logger(logging_level=logging.INFO):
    # Define Logger
    logger = logging.getLogger()

    # Set Logger Display Level
    logger.setLevel(level=logging_level)

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    )
    logger.addHandler(stream_handler)

    return logger


class RESULT_OBJECT(object):
    def __init__(self, seq_name, logger):
        # Set Sequence Name
        self.seq_name = seq_name

        # Video File
        self.video_file_path = None

        # Image Files Path
        self.img_files_path = None

        # Set Logger
        self.logger = logger

    def __repr__(self):
        return self.seq_name

    def __len__(self):
        return len(self.img_files_path)

    def get_data(self, img_files_base_path, video_file_path):
        self.video_file_path = video_file_path
        self.img_files_path = img_files_base_path

    def pack(self, dst_dir, imgs_sampling_fidx_interval=1):
        _now = datetime.datetime.now()
        zip_file_path = os.path.join(
            dst_dir, "{}.zip".format(self)
        )

        with ZipFile(zip_file_path, "w") as z_f:
            # Log Msg
            log_msg = "[PACK] Making {} zip file...!".format(self)
            self.logger.info(log_msg)

            # Add Video File to Zip File
            z_f.write(self.video_file_path, self.video_file_path.split("/")[-1])

            # Add Image Files to Zip File
            img_files_zip_base_path = self.img_files_path.split("/")[-1]
            img_fnames = sorted(os.listdir(self.img_files_path))
            img_files_path = [
                os.path.join(self.img_files_path, fm) for fm in img_fnames
            ]
            for jj, img_file_path in enumerate(img_files_path):
                if jj % imgs_sampling_fidx_interval == 0:
                    z_f.write(
                        img_file_path, os.path.join(img_files_zip_base_path, img_fnames[jj])
                    )


if __name__ == "__main__":
    # Set Logger
    logger = set_logger()

    # Set Base Path

    _fnames_ = os.listdir(__BASE_PATH__)

    # Destination Path
    __DST_PATH__ = "/home/kyle/sendings"

    # Detect AVI Files (video)
    video_fnames = [fn for fn in _fnames_ if fn.endswith("avi")]
    video_files_path = [os.path.join(__BASE_PATH__, fnn) for fnn in video_fnames]

    # Split Directories
    _split_result_imgs_base_dirs_ = [
        os.path.join(__BASE_PATH__, d, "results") for d in _fnames_ if os.path.isdir(os.path.join(__BASE_PATH__, d))
    ]

    # Sequence Base Dirs
    seq_result_imgs_path_list = []
    for _fn in _split_result_imgs_base_dirs_:
        _fn2 = os.listdir(_fn)
        for _fn2_j in _fn2:
            seq_result_imgs_path = os.path.join(_fn, _fn2_j)
            if os.path.isdir(seq_result_imgs_path):
                seq_result_imgs_path_list.append(seq_result_imgs_path)

    # Init Objects
    result_objs = []
    for j, video_fname in enumerate(video_fnames):
        seq_name = video_fname.split(".")[0]

        result_obj = RESULT_OBJECT(seq_name=seq_name, logger=logger)

        result_obj.get_data(
            img_files_base_path=seq_result_imgs_path_list[j],
            video_file_path=video_files_path[j]
        )

        result_objs.append(result_obj)

    # Pack Objects
    for result_obj in result_objs:
        result_obj.pack(dst_dir=__DST_PATH__, imgs_sampling_fidx_interval=50)
