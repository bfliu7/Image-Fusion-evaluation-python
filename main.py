import os
import numpy as np
import cv2
from os.path import join, exists
import argparse
from compute_metric import __metrics__
import pandas as pd
from openpyxl.utils import get_column_letter
import shutil

Parse = argparse.ArgumentParser()
Parse.add_argument("--VIFB_PATH", default="dataset", required=True, help="the path to dataset folder")
Parse.add_argument("--bench", default="40_vot_tno", required=True, help="choose the benchmark dataset")
Parse.add_argument("--fuse_vi_channel", default="3", required=True, help="num of bench's fuse_vi_channel")
Parse.add_argument("--ir_channel", default="1", required=True, help="num of bench's infrared channel")
Parse.add_argument("--method", default="tefuse", required=True, help="choose the method to test")

args = Parse.parse_args()


def read_img(image_name, channel):
    if channel == "3":
        image = cv2.imread(image_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float64)
        return image_rgb
    else:  # channel == 1
        image_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        image_gray = image_gray[:, :, np.newaxis]
        image_gray = image_gray.astype(np.float64)

        return image_gray


def save_to_excel(data, path):
    output_xlsx = join(path, "output_single.xlsx")
    standard_column_width = 20
    standard_row_height = 20
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        data.to_excel(writer, index=False)
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        for col in worksheet.columns:
            max_length = 0
            column = get_column_letter(col[0].column)
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            worksheet.column_dimensions[column].width = max(max_length * 1.2, standard_column_width)

        for row in worksheet.rows:
            for cell in row:
                worksheet.row_dimensions[cell.row].height = standard_row_height

    workbook.save(output_xlsx)


def ensure_dir(directory):
    if not exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)


def test(vifb_path, bench, method, fuse_vi_channel, ir_channel):
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Now processing {} with method {}.".format(bench, method))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # get the path
    # abs_path = os.path.dirname(os.path.abspath(__file__))
    # metrics_method_path = join(abs_path, 'compute_metric')

    fused_path = join(vifb_path, bench, method)
    visible_path = join(vifb_path, bench, 'vis')
    infrared_path = join(vifb_path, bench, 'ir')

    output_path = join(vifb_path, 'output', bench, method, 'evaluation_metrics')
    output_path_single = join(vifb_path, 'output', bench, method, 'evaluation_metrics_single')

    # check
    assert not (fuse_vi_channel is "1" and ir_channel is "3")  # ensure fuse_channel is 1 while ir_channel is not 3
    assert exists(vifb_path) and exists(fused_path) and exists(visible_path) and exists(infrared_path)

    ensure_dir(output_path)
    ensure_dir(output_path_single)

    # VI, IR and fuse triple
    imgs_triple = []
    for image_vi in os.listdir(visible_path):
        for image_ir in os.listdir(infrared_path):
            for image_fuse in os.listdir(fused_path):
                image_vi_ = image_vi[3:]  # ignore "VIS"
                image_ir_ = image_ir[2:]  # ignore "IR"
                image_fuse_ = image_fuse[4:]  # ignore "Fuse"
                if image_vi_ == image_ir_ == image_fuse_:
                    imgs_triple.append((image_vi, image_ir, image_fuse))
                    break
    # store as excel
    # compute the metrics
    df = pd.DataFrame()
    image_names = []
    for img_triple in imgs_triple:
        image_names.append(str(img_triple[2]).split('.')[0])

    for name, metric_method in __metrics__.items():
        print("Using metric {}.".format(name))
        results = []
        for img_vi_path, img_ir_path, img_fuse_path in imgs_triple:
            image_name = str(img_fuse_path).split(".")[0] + '.txt'

            img_vi_path = join(visible_path, img_vi_path)
            img_ir_path = join(infrared_path, img_ir_path)
            img_fuse_path = join(fused_path, img_fuse_path)

            img_vi = read_img(img_vi_path, fuse_vi_channel)
            img_ir = read_img(img_ir_path, ir_channel)
            img_fuse = read_img(img_fuse_path, fuse_vi_channel)

            single_result = metric_method(img_vi, img_ir, img_fuse)
            results.append(single_result)

            # store one single result
            # store like .txt
            store_single_result = join(output_path_single, image_name)
            with open(store_single_result, 'a') as single_f:
                single_f.write("{}:{}\n".format(name, str(single_result)))

        df[name] = results

        result_mean = sum(results) / len(results)
        print("    {} is : {} ".format(name, str(result_mean)))

        # store all results in .txt
        store_all_results = join(output_path, "all_results.txt")
        with open(store_all_results, 'a') as all_f:
            all_f.write("{}:{}\n".format(name, str(result_mean)))

    # store like .xls
    df.insert(loc=0, column=" ", value=image_names)

    save_to_excel(df, output_path_single)


if __name__ == "__main__":
    VIFB_PATH = args.VIFB_PATH
    bench = args.bench
    method = args.method
    fuse_vi_channel = args.fuse_vi_channel
    ir_channel = args.ir_channel
    test(VIFB_PATH, bench, method, fuse_vi_channel, ir_channel)
