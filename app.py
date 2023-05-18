from flask import Flask, render_template, request, redirect, url_for, Response
import numpy as np
from DeepImageSearch import Load_Data, Search_Setup
from PIL import Image
import os
from werkzeug.utils import secure_filename
import random
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["POST","GET"])
def add():
    if request.method == "POST":
        photo = request.files["input_image"]
        filename = secure_filename(photo.filename) # save file 
        filepath = os.path.join('input_file', filename)
        photo.save(filepath)

        folder = 'input_file'
        file_path = os.path.join(folder,filename)

        dl = Load_Data()
        image_list = dl.from_folder(["./static/images"])

        if 'metadata-files' in os.listdir():
            if os.listdir('metadata-files') != 0:
                st = Search_Setup(image_list, model_name="vgg19", pretrained=True, image_count=None)
                st.run_index(b='no')
            else:
                st = Search_Setup(image_list, model_name="vgg19", pretrained=True, image_count=None)
                st.run_index(b='yes')
        else:
            st = Search_Setup(image_list, model_name="vgg19", pretrained=True, image_count=None)
            st.run_index(b='yes')

        folder = 'input_file'
        file_path = os.path.join(folder,filename)
        
        st.add_images_to_index([file_path])

        final_image = st.get_similar_images(image_path=file_path, number_of_images=9)

        curnt_image = []
        for id1 in final_image.values():
            curnt_image.append(id1.split('/')[-1])

        df = pd.read_csv('fashion.csv')

        new_list = []
        for j in range(len(df)):
            for i in curnt_image:
                if df['Image'][j] == i:
                    my_dict={'Input_Image':i,
                        'Gender':df['Gender'][j],
                        'Title':df['ProductTitle'][j],
                        'Category':df['Category'][j],
                        'Sub_Category':df['SubCategory'][j],
                        'Product_Type':df['ProductType'][j]}
                    new_list.append(my_dict)   
        
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

        return render_template("index.html", data = new_list)  
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)