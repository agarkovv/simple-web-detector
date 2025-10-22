import json
import os
import statistics

import pytest
import requests
from furl import furl


@pytest.fixture(scope="session")
def eval_data():
    with open("eval.json", "r") as f:
        return json.loads(f.read())


@pytest.fixture(scope="session")
def server_ip():
    # server_ip_value = os.environ['DOCKER_IP']
    server_ip_value = "localhost"
    if server_ip_value is None:
        pytest.fail("No cluster ip were provided")
    return server_ip_value


@pytest.fixture(scope="session")
def http_host(server_ip):
    return "http://{}:8080/".format(server_ip)


def get_image_link(image_name):
    return "http://images.cocodataset.org/val2017/{}".format(image_name)


def calc_score(actual, predicted):
    actual_copy = [x for x in actual]
    score = 0
    for label in predicted:
        if label in actual_copy:
            score += 1
            actual_copy.remove(label)
    return 2 * score / (len(actual) + len(predicted))


@pytest.mark.run(order=2)
def test_http_endpoint(http_host, eval_data, capsys):
    with capsys.disabled():
        predict_url = str(furl(http_host) / "predict")
        scores = []
        for img_name, labels in eval_data.items():
            img_url = get_image_link(img_name)
            r = requests.post(predict_url, json={"url": img_url})
            predicted_labels = r.json()["objects"]
            scores.append(calc_score(labels, predicted_labels))

        mean_score = statistics.mean(scores)
        assert mean_score > 0.5
