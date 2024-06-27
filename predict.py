import json
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse


class Utils:

    def __init__(self):
        pass

    def parseScriptParams(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('imagePath')
        parser.add_argument('model')
        parser.add_argument('--top_k', default=5)
        parser.add_argument('--category_names', default='label_map.json')
        args = parser.parse_args()
        return args

    def readLabelsMap(self, label_map_file_name):
        with open(label_map_file_name, 'r') as f:
            class_names = json.load(f)
        return class_names

    def process_image(self, image, image_size=224):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))
        image /= 255.0
        return image.numpy()

    def load_model(self, model_filepath):
        reloaded_keras_model = tf.keras.models.load_model(model_filepath, custom_objects={'KerasLayer': hub.KerasLayer})
        return reloaded_keras_model

    def predict(self, image_path, model, labelsMap, top_k=5):
        im = Image.open(image_path)
        test_image = np.asarray(im)
        processed_test_image = self.process_image(test_image)
        processed_test_image = np.expand_dims(processed_test_image, axis=0)
        ps = model.predict(processed_test_image, verbose=None)
        my_dict = {index: value for index, value in enumerate(ps[0])}
        # print(my_dict)
        top_k_elements = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True)[:top_k])
        return list(top_k_elements.values()), [labelsMap[str(i)] for i in top_k_elements.keys()]


if __name__ == "__main__":
    util = Utils()
    # Parse Script Arguments
    arguments = util.parseScriptParams()
    ImagePath = arguments.imagePath
    model_name = arguments.model
    top_k = int(arguments.top_k)
    category_names = arguments.category_names
    # Load Model
    model = util.load_model(model_name)
    #Load Labels
    labels = util.readLabelsMap(category_names)
    # Predict Image
    props, classes = util.predict(ImagePath, model, labels, top_k)
    # print result
    # print(props)
    # print(classes)
    print("\n{:<30} {:<35}".format('Class', 'Probability'))
    print("------------------------------------------")
    for i in range(len(props)):
        print("{:<30} {:<35}".format(classes[i], props[i]))

    max_value = max(props)
    max_index = props.index(max_value)
    print("\n**The result is {}".format(classes[max_index]))

# To do
# design output as Table