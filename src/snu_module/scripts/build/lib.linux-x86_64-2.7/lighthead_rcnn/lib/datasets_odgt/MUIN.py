# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

class MUINBasic:
    class_names = [
        'background', 'person', 'lie_person']
    classes_originID = {
        'person': 1, 'lie_person': 2}
    num_classes = 3


class MUIN(MUINBasic):
    pass
    # train_root_folder = ''
    # train_source = os.path.join(
    #     config.root_dir, 'data', 'MSCOCO/odformat/coco_trainvalmini.odgt')
    # eval_root_folder = ''
    # eval_source = os.path.join(
    #     config.root_dir, 'data', 'MSCOCO/odformat/coco_minival2014.odgt')
    # eval_json = os.path.join(
    #     config.root_dir, 'data', 'MSCOCO/instances_minival2014.json')


if __name__ == "__main__":
    # coco = COCOIns()
    from IPython import embed
    embed()