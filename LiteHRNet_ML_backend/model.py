from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase


from LiteHRNet_ONNX import LiteHRNet_ONNX_Model

predictor = LiteHRNet_ONNX_Model()


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def format_response(self, keypoints, resized_shape, original_shape) -> list:
        results = []
        keypoints[:, :2] = keypoints[:, :2] / resized_shape[::-1]
        original_height, original_width = original_shape[0], original_shape[1]
        keypoint_mapping = ["Nose",
                            "Left Eye",
                            "Right Eye",
                            "Left Ear",
                            "Right Ear",
                            "Left Shoulder",
                            "Right Shoulder",
                            "Left Elbow",
                            "Right Elbow",
                            "Left Wrist",
                            "Right Wrist",
                            "Left Hip",
                            "Right Hip",
                            "Left Knee",
                            "Right Knee",
                            "Left Ankle",
                            "Right Ankle"]
        for i, keypoint in enumerate(keypoints):
            results.append({
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    "x": float(keypoint[0] * 100),
                    "y": float(keypoint[1] * 100),
                    "width": 0.18711634418638043,
                    "keypointlabels": [keypoint_mapping[i]]
                },
                "from_name": "kp-1",
                "to_name": "img-1",
                "type": "keypointlabels",
                "origin": "manual"
            })
        return results

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs):
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]

        img_path = tasks[0]['data']['img']
        task_id = tasks[0]['id']

        keypoints, resized_shape, origin_shape = predictor.predict(
            img_path=img_path, task_id=task_id)
        results = self.format_response(keypoints, resized_shape, origin_shape)
        # results = [
        #     {
        #         "from_name": "kp-1",
        #         "image_rotation": 0,
        #         "origin": "manual",
        #         "original_height": 417,
        #         "original_width": 626,
        #         "to_name": "img-1",
        #         "type": "keypointlabels",
        #         "value": {
        #             "keypointlabels": [
        #                 "Right Wrist"
        #             ],
        #             "width": 0.18711634418638043,
        #             "x": 50,
        #             "y": 50
        #         }
        #     },
        #     {
        #         "from_name": "kp-1",
        #         "image_rotation": 0,
        #         "origin": "manual",
        #         "original_height": 417,
        #         "original_width": 626,
        #         "to_name": "img-1",
        #         "type": "keypointlabels",
        #         "value": {
        #             "keypointlabels": [
        #                 "Right Wrist"
        #             ],
        #             "width": 0.18711634418638043,
        #             "x": 70.234,
        #             "y": 70.567
        #         }
        #     },
        #     results_[0]
        # ]

        return [{
            'result': results,
            'model_version': self.get('model_version')
        }]

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
