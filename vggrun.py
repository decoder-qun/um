{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "1228night.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyMPd74vp+VkYTyzLEuEkC+L",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/decoder-qun/um/blob/master/vggrun.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "byKmRyNItbhi",
        "outputId": "17f872a3-b963-4cc3-ad3a-2ceaa33d3a51"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /gdrive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/gdrive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "torch.cuda.empty_cache()"
      ],
      "metadata": {
        "id": "TVZEBkAv54kE"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "%cd /gdrive/MyDrive/Colab Notebooks/um1/um\n",
        "!python vggrun.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HKmOFvUytv5Q",
        "outputId": "1167395e-0c8a-455d-c708-5171d911194f"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/gdrive/MyDrive/Colab Notebooks/um1/um\n",
            "cuda\n",
            "Dataset:multi\tSource:real\tTarget:sketch\tLabeled num perclass:3\tNetwork:vgg\t\n",
            "126 classes in this dataset\n",
            "vggrun.py:128: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.\n",
            "  source_labeled_image = Variable(source_labeled_image,volatile=True)\n",
            "vggrun.py:129: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.\n",
            "  source_labeled_label = Variable(source_labeled_label,volatile=True)\n",
            "vggrun.py:130: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.\n",
            "  target_unlabeled_image = Variable(target_unlabeled_image,volatile=True)\n",
            "vggrun.py:131: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.\n",
            "  target_unlabeled_image2 = Variable(target_unlabeled_image2,volatile=True)\n",
            "vggrun.py:132: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.\n",
            "  target_labeled_image = Variable(target_labeled_image,volatile=True)\n",
            "vggrun.py:133: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.\n",
            "  target_labeled_label = Variable(target_labeled_label,volatile=True)\n",
            "vggrun.py:134: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.\n",
            "  val_image = Variable(val_image,volatile=True)\n",
            "vggrun.py:135: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.\n",
            "  val_label = Variable(val_label,volatile=True)\n",
            "=> no checkpoint found at 'True'\n",
            "/gdrive/MyDrive/Colab Notebooks/um1/um/loss.py:6: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
            "  predict=F.softmax(predict)\n",
            "Traceback (most recent call last):\n",
            "  File \"vggrun.py\", line 528, in <module>\n",
            "    \n",
            "  File \"vggrun.py\", line 197, in main\n",
            "    source_labeled = next(source_labeled_iter)\n",
            "  File \"/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py\", line 521, in __next__\n",
            "    data = self._next_data()\n",
            "  File \"/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py\", line 1186, in _next_data\n",
            "    idx, data = self._get_data()\n",
            "  File \"/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py\", line 1152, in _get_data\n",
            "    success, data = self._try_get_data()\n",
            "  File \"/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py\", line 990, in _try_get_data\n",
            "    data = self._data_queue.get(timeout=timeout)\n",
            "  File \"/usr/lib/python3.7/multiprocessing/queues.py\", line 104, in get\n",
            "    if not self._poll(timeout):\n",
            "  File \"/usr/lib/python3.7/multiprocessing/connection.py\", line 257, in poll\n",
            "    return self._poll(timeout)\n",
            "  File \"/usr/lib/python3.7/multiprocessing/connection.py\", line 414, in _poll\n",
            "    r = wait([self], timeout)\n",
            "  File \"/usr/lib/python3.7/multiprocessing/connection.py\", line 921, in wait\n",
            "    ready = selector.select(timeout)\n",
            "  File \"/usr/lib/python3.7/selectors.py\", line 415, in select\n",
            "    fd_event_list = self._selector.poll(timeout)\n",
            "KeyboardInterrupt\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import gc\n",
        "\n",
        "del \n",
        "gc.collect()"
      ],
      "metadata": {
        "id": "VKVW0m6E2zcM",
        "outputId": "5a15f093-3e26-4fac-ca58-8cf3ceb9922b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 222
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-9-80bfa73e9802>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mgc\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0;32mdel\u001b[0m \u001b[0mall\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0mgc\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcollect\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'all' is not defined"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%%html\n",
        "<link rel=\"stylesheet\" href=\"/nbextensions/google.colab/tabbar.css\">\n",
        "<div class='goog-tab'>\n",
        "  Some content\n",
        "</div>"
      ],
      "metadata": {
        "id": "h22OajOn2iWW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import portpicker\n",
        "import threading\n",
        "import socket\n",
        "import IPython\n",
        "\n",
        "from six.moves import socketserver\n",
        "from six.moves import SimpleHTTPServer\n",
        "\n",
        "class V6Server(socketserver.TCPServer):\n",
        "  address_family = socket.AF_INET6\n",
        "\n",
        "class Handler(SimpleHTTPServer.SimpleHTTPRequestHandler):\n",
        "  def do_GET(self):\n",
        "    self.send_response(200)\n",
        "    # If the response should not be cached in the notebook for\n",
        "    # offline access:\n",
        "    # self.send_header('x-colab-notebook-cache-control', 'no-cache')\n",
        "    self.end_headers()\n",
        "    self.wfile.write(b'''\n",
        "      document.querySelector('#output-area').appendChild(document.createTextNode('Script result!'));\n",
        "    ''')\n",
        "\n",
        "port = portpicker.pick_unused_port()\n",
        "\n",
        "def server_entry():\n",
        "    httpd = V6Server(('::', port), Handler)\n",
        "    # Handle a single request then exit the thread.\n",
        "    httpd.serve_forever()\n",
        "\n",
        "thread = threading.Thread(target=server_entry)\n",
        "thread.start()\n",
        "\n",
        "# Display some HTML referencing the resource.\n",
        "display(IPython.display.HTML('<script src=\"https://localhost:{port}/\"></script>'.format(port=port)))"
      ],
      "metadata": {
        "id": "1K2Ke0Vt2iWf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import output\n",
        "output.serve_kernel_port_as_iframe(port)"
      ],
      "metadata": {
        "id": "-md1vv8b2iWf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import output\n",
        "output.serve_kernel_port_as_window(port)"
      ],
      "metadata": {
        "id": "fjEWLjsz2iWg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "torch.cuda.empty_cache()\n",
        "torch.cuda.memory_summary(device=None, abbreviated=False)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 192
        },
        "id": "-GPgUmtBucrH",
        "outputId": "6c38defb-5312-4db0-8edc-790428d996bb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'|===========================================================================|\\n|                  PyTorch CUDA memory summary, device ID 0                 |\\n|---------------------------------------------------------------------------|\\n|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |\\n|===========================================================================|\\n|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |\\n|---------------------------------------------------------------------------|\\n| Allocated memory      |       0 B  |       0 B  |       0 B  |       0 B  |\\n|       from large pool |       0 B  |       0 B  |       0 B  |       0 B  |\\n|       from small pool |       0 B  |       0 B  |       0 B  |       0 B  |\\n|---------------------------------------------------------------------------|\\n| Active memory         |       0 B  |       0 B  |       0 B  |       0 B  |\\n|       from large pool |       0 B  |       0 B  |       0 B  |       0 B  |\\n|       from small pool |       0 B  |       0 B  |       0 B  |       0 B  |\\n|---------------------------------------------------------------------------|\\n| GPU reserved memory   |       0 B  |       0 B  |       0 B  |       0 B  |\\n|       from large pool |       0 B  |       0 B  |       0 B  |       0 B  |\\n|       from small pool |       0 B  |       0 B  |       0 B  |       0 B  |\\n|---------------------------------------------------------------------------|\\n| Non-releasable memory |       0 B  |       0 B  |       0 B  |       0 B  |\\n|       from large pool |       0 B  |       0 B  |       0 B  |       0 B  |\\n|       from small pool |       0 B  |       0 B  |       0 B  |       0 B  |\\n|---------------------------------------------------------------------------|\\n| Allocations           |       0    |       0    |       0    |       0    |\\n|       from large pool |       0    |       0    |       0    |       0    |\\n|       from small pool |       0    |       0    |       0    |       0    |\\n|---------------------------------------------------------------------------|\\n| Active allocs         |       0    |       0    |       0    |       0    |\\n|       from large pool |       0    |       0    |       0    |       0    |\\n|       from small pool |       0    |       0    |       0    |       0    |\\n|---------------------------------------------------------------------------|\\n| GPU reserved segments |       0    |       0    |       0    |       0    |\\n|       from large pool |       0    |       0    |       0    |       0    |\\n|       from small pool |       0    |       0    |       0    |       0    |\\n|---------------------------------------------------------------------------|\\n| Non-releasable allocs |       0    |       0    |       0    |       0    |\\n|       from large pool |       0    |       0    |       0    |       0    |\\n|       from small pool |       0    |       0    |       0    |       0    |\\n|---------------------------------------------------------------------------|\\n| Oversize allocations  |       0    |       0    |       0    |       0    |\\n|---------------------------------------------------------------------------|\\n| Oversize GPU segments |       0    |       0    |       0    |       0    |\\n|===========================================================================|\\n'"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        ""
      ],
      "metadata": {
        "id": "pzaqYgvyxHn0"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "%tensorflow_version 1.x\n",
        "import tensorflow as tf\n",
        "import timeit\n",
        " \n",
        "device_name = tf.test.gpu_device_name()\n",
        "if device_name != '/device:GPU:0':\n",
        "  print(\n",
        "      '\\n\\nThis error most likely means that this notebook is not '\n",
        "      'configured to use a GPU.  Change this in Notebook Settings via the '\n",
        "      'command palette (cmd/ctrl-shift-P) or the Edit menu.\\n\\n')\n",
        "  raise SystemError('GPU device not found')\n",
        " \n",
        "def cpu():\n",
        "  with tf.device('/cpu:0'):\n",
        "    random_image_cpu = tf.random.normal((100, 100, 100, 3))\n",
        "    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)\n",
        "    return tf.math.reduce_sum(net_cpu)\n",
        " \n",
        "def gpu():\n",
        "  with tf.device('/device:GPU:0'):\n",
        "    random_image_gpu = tf.random.normal((100, 100, 100, 3))\n",
        "    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)\n",
        "    return tf.math.reduce_sum(net_gpu)\n",
        "  \n",
        "# We run each op once to warm up; see: https://stackoverflow.com/a/45067900\n",
        "cpu()\n",
        "gpu()\n",
        " \n",
        "# Run the op several times.\n",
        "print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '\n",
        "      '(batch x height x width x channel). Sum of ten runs.')\n",
        "print('CPU (s):')\n",
        "cpu_time = timeit.timeit('cpu()', number=10, setup=\"from __main__ import cpu\")\n",
        "print(cpu_time)\n",
        "print('GPU (s):')\n",
        "gpu_time = timeit.timeit('gpu()', number=10, setup=\"from __main__ import gpu\")\n",
        "print(gpu_time)\n",
        "print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mJW83u5Xvms-",
        "outputId": "b71255e4-d334-416c-d404-65523b92de7e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "TensorFlow 1.x selected.\n",
            "WARNING:tensorflow:From /tensorflow-1.15.2/python3.7/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "If using Keras pass *_constraint arguments to layers.\n",
            "Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images (batch x height x width x channel). Sum of ten runs.\n",
            "CPU (s):\n",
            "0.1754167179999513\n",
            "GPU (s):\n",
            "0.18236020600011216\n",
            "GPU speedup over CPU: 0x\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!kill -9 [pid]"
      ],
      "metadata": {
        "id": "CIg3ctL6vpH_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!nvidia-sim"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "x2zZnwx5wGWA",
        "outputId": "8d4e3f61-b0e3-43a0-cc22-946eeb41ee3c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/bin/bash: nvidia-sim: command not found\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!kill -9 -1"
      ],
      "metadata": {
        "id": "mtNjPApDwKSE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!fuser -v /dev/nvidia*"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HbdcOgYiwrnz",
        "outputId": "3b0606a5-50d0-4fa8-a67a-a83e97f883b8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Specified filename /dev/nvidia* does not exist.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "0FoG72nEwzbZ"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        ""
      ],
      "metadata": {
        "id": "u1StWsSj1CE-"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}