/* eslint "no-global-assign": ["error", {"exceptions": [ "TextDecoder", "TextEncoder", "URLSearchParams" ] } ] */
/* global view */

var host = {};
let lastAddedOperatorName = null;

function showWaitPage(isWait) {
  if (isWait) {
    document.body.classList.remove("default")
    document.body.classList.add("welcome")
    document.body.classList.add("spinner")
  } else {
    document.body.classList.add("default")
    document.body.classList.remove("welcome")
    document.body.classList.remove("spinner")
  }
}

host.BrowserHost = class {
  constructor() {
    this._random_session = Math.random();
    this._index_session = '';
    this._document = window.document;
    this._window = window;
    this._navigator = navigator;
    this._window.eval = () => {
      throw new Error('window.eval() not supported.');
    };
    this._meta = {};
    for (const element of Array.from(this._document.getElementsByTagName('meta'))) {
      if (element.content) {
        this._meta[element.name] = this._meta[element.name] || [];
        this._meta[element.name].push(element.content);
      }
    }
    this._type = this._meta.type ? this._meta.type[0] : 'Browser';
    this._version = this._meta.version ? this._meta.version[0] : null;
    this._telemetry = this._version && this._version !== '0.0.0';
    this._environment = new Map();
    this._environment.set('zoom', 'scroll');
    // this._environment.set('zoom', 'drag');
    this._ori_model_file = null;
    this._activate_model_file = null;
    this._modify_info = [];
    window._host = this;
  }

  get session() {
    return `${this._index_session}${this._random_session}`;
  }

  get window() {
    return this._window;
  }

  get document() {
    return this._document;
  }

  get version() {
    return this._version;
  }

  get type() {
    return this._type;
  }

  get agent() {
    const userAgent = this._navigator.userAgent.toLowerCase();
    if (userAgent.indexOf('safari') !== -1 && userAgent.indexOf('chrome') === -1) {
      return 'safari';
    }
    return 'any';
  }

  initialize(view) {
    this._view = view;
    this.view = view;
    return new Promise((resolve) => {
      resolve();
    });
  }

  start() {
    this.window.addEventListener('error', (e) => {
      this.exception(e.error, true);
    });

    const params = new URLSearchParams(this.window.location.search);
    this._environment.set('zoom', params.has('zoom') ? params.get('zoom') : this._environment.get('zoom'));

    this._menu = new host.Dropdown(this, 'menu-button', 'menu-dropdown');
    this._menu.add({
      label: 'Open New Window...',
      accelerator: 'CmdOrCtrl+O',
      click: () => {
        if (this.window.is_electron) {
          this.window.new_window();
        } else {
          this.window.open('/');
        }
      },
    });

    this._menu.add({});
    this._menu.add({
      label: 'Undo',
      accelerator: 'CmdOrCtrl+Z',
      click: () => this._view.modifier.undo(),
    });
    this._menu.add({
      label: 'Redo',
      accelerator: 'CmdOrCtrl+Y',
      click: () => this._view.modifier.redo(),
    });
    this._menu.add({
      label: 'Properties...',
      accelerator: 'CmdOrCtrl+Enter',
      click: () => this._view.showModelProperties(),
    });
    this._menu.add({});
    this._menu.add({
      label: 'Find...',
      accelerator: 'CmdOrCtrl+F',
      click: () => this._view.find(),
    });
    this._menu.add({});
    this._menu.add({
      label: () => (this._view.options.attributes ? 'Hide Attributes' : 'Show Attributes'),
      accelerator: 'CmdOrCtrl+D',
      click: () => this._view.toggle('attributes'),
    });
    this._menu.add({
      label: () => (this._view.options.initializers ? 'Hide Initializers' : 'Show Initializers'),
      accelerator: 'CmdOrCtrl+I',
      click: () => this._view.toggle('initializers'),
    });
    this._menu.add({
      label: () => (this._view.options.names ? 'Hide Names' : 'Show Names'),
      accelerator: 'CmdOrCtrl+U',
      click: () => this._view.toggle('names'),
    });
    this._menu.add({
      label: () => (this._view.options.direction === 'vertical' ? 'Show Horizontal' : 'Show Vertical'),
      accelerator: 'CmdOrCtrl+K',
      click: () => this._view.toggle('direction'),
    });
    this._menu.add({
      label: () => (this._view.options.mousewheel === 'scroll' ? 'Mouse Wheel: Zoom' : 'Mouse Wheel: Scroll'),
      accelerator: 'CmdOrCtrl+M',
      click: () => this._view.toggle('mousewheel'),
    });
    this._menu.add({});
    this._menu.add({
      label: 'Zoom In',
      accelerator: 'Shift+Up',
      click: () => this.document.getElementById('zoom-in-button').click(),
    });
    this._menu.add({
      label: 'Zoom Out',
      accelerator: 'Shift+Down',
      click: () => this.document.getElementById('zoom-out-button').click(),
    });
    this._menu.add({
      label: 'Actual Size',
      accelerator: 'Shift+Backspace',
      click: () => this._view.resetZoom(),
    });
    this._menu.add({});
    this._menu.add({
      label: 'Export as PNG',
      accelerator: 'CmdOrCtrl+Shift+E',
      click: () => this._view.export(document.title + '.png'),
    });
    this._menu.add({
      label: 'Export as SVG',
      accelerator: 'CmdOrCtrl+Alt+E',
      click: () => this._view.export(document.title + '.svg'),
    });
    this.document.getElementById('menu-button').addEventListener('click', (e) => {
      this._menu.toggle();
      e.preventDefault();
    });
    this._menu.add({});
    this._menu.add({
      label: 'About ' + this.document.title,
      click: () => this._about(),
    });

    this.document.getElementById('modify-export').addEventListener('click', () => {
      let export_name = 'modify_info.json';
      if (this._ori_model_file) {
        export_name = `${this._ori_model_file.name}.${export_name}`;
      }
      let modify_info = [...this._modify_info, { path: '/download', data_body: this.build_download_data(true) }];
      this.export(export_name, new Blob([JSON.stringify(modify_info)], { type: 'text/plain' }));
    });

    this.document.getElementById('modify-import').addEventListener('click', () => {
      this.document.getElementById('open-modify-json-dialog').click();
    });

    this.init_input_shape_change_event();
    this.init_batch_size_change_event();

    const openJsonFileDialog = this.document.getElementById('open-modify-json-dialog');

    openJsonFileDialog.addEventListener('change', (e) => {
      if (e.target && e.target.files && e.target.files.length > 0) {
        const files = Array.from(e.target.files);
        const file = files[0];
        let reader = new FileReader();
        reader.onload = async () => {
          let modify_infos = JSON.parse(reader.result);

          this.take_effect_modify('/load-json', { modify_infos, session: this.session }, true, null, "Load Json Success");
        };
        reader.readAsText(file);
        openJsonFileDialog.value = null;
      }
    });

    const resetButton = this.document.getElementById('reset-graph');
    resetButton.addEventListener('click', () => {
      if (window.DISPLAY_OM_MODEL) {
        this.show_alert_message('disabled', 'This button is disabled when displaying om model.');
        return;
      }
      // this._view._graph.resetGraph();
      // this._view._updateGraph();
      this.confirm('Comfirm', 'are you sure to reset? All modifications cannot be reverted').then((confirmed) => {
        if (!confirmed) {
          return;
        }
        if (this._ori_model_file !== this._activate_model_file && this._ori_model_file != null) {
          this.openFile(this._ori_model_file);
        } else {
          this._view.modifier.resetGraph();
        }
        this._modify_info = [];
      });
    });

    const downloadWithShapeInfCheckBox = this.document.getElementById('shapeInference');
    downloadWithShapeInfCheckBox.addEventListener('click', () => {
      this._view.modifier.onOffShapeInf(downloadWithShapeInfCheckBox.checked);
    });
    const downloadWithCleanUp = this.document.getElementById('cleanUp');
    downloadWithCleanUp.addEventListener('click', () => {
      this._view.modifier.onOffCleanUp(downloadWithCleanUp.checked);
    });

    const downloadButton = this.document.getElementById('download-graph');

    if (this.window.is_electron && this.window.fetch_electron) {
      class Response {
        constructor(status, msg, file) {
          this._status = status;
          this._msg = msg;
          this._file = file;
          this._headers = new Map();
          this._headers.set('Content-Disposition', `filename=${this._msg.filepath}`);
        }

        get headers() {
          return this._headers;
        }

        text() {
          return Promise.resolve(this._msg);
        }

        blob() {
          if (!this._file) {
            return Promise.resolve(null);
          }
          let blob = new Blob([this._file]);
          blob.filepath = this._msg.filepath;
          return Promise.resolve(blob);
        }

        get status() {
          return this._status;
        }

        get ok() {
          return 200 <= this._status && this._status < 300;
        }
      }
      this.window.fetch = (path, options) => {
        let body = options.body;
        if (body instanceof FormData) {
          let body_obj = {};
          for (const [key, value] of body.entries()) {
            if (value instanceof File) {
              body_obj[key] = value.path || value.filepath;
            } else {
              body_obj[key] = value;
            }
          }
          body = body_obj;
        } else if (typeof body == 'string') {
          body = JSON.parse(body);
        }
        return fetch_electron(path, body).then((result) => {
          let [status, msg, file] = result;
          return new Response(status, msg, file);
        });
      };
    }

    fetch('/get_session_index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session: this.session }),
    })
      .then((response) => {
        this.check_res_status(response.status);
        return response.text().then((text) => {
          this._index_session = text;
        });
      })
      .then(() => {
        fetch('/init', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session: this.session }),
        }).then((response) => {
          this.check_res_status(response.status);
          response.blob().then((blob) => {
            if (blob && blob.size > 0) {
              let disposition = response.headers.get('Content-Disposition');
              if (disposition) {
                this.upload_filepath = disposition.substring(disposition.indexOf('=') + 1);
                let last_index = disposition.replace(/[\\\\\/]/g, '#').lastIndexOf('#');
                if (last_index >= 0) {
                  this.upload_filename = disposition.substring(last_index + 1);
                } else {
                  this.upload_filename = this.upload_filepath;
                }
              } else {
                this.upload_filepath = 'some.onnx';
                this.upload_filename = 'some.onnx';
              }
              let file = new File([blob], this.upload_filename);
              this._ori_model_file = file;
              file.filepath = blob.filepath ? blob.filepath : this.upload_filepath;
              return this.openFile(file);
            }
          });
        });
      });

    downloadButton.addEventListener('click', () => {
      if (window.DISPLAY_OM_MODEL) {
        this.show_alert_message('disabled', 'This button is disabled when displaying om model.');
        return;
      }
      let dialog = this.document.getElementById('download-dialog');
      this.show_confirm_dialog(dialog).then((is_not_cancel) => {
        if (!is_not_cancel) {
          return;
        }

        this.take_effect_modify('/download', this.build_download_data(true), false, (blob) => {
          this.export(this.upload_filename, blob);
          this.show_message('Success!', 'Model has been successfuly modified', 'success');
        });
      });
    });

    const onnxSimButton = this.document.getElementById('onnxsim-graph');
    onnxSimButton.addEventListener('click', () => {
      if (window.DISPLAY_OM_MODEL) {
        this.show_alert_message('disabled', 'This button is disabled when displaying om model.');
        return;
      }
      this.take_effect_modify('/onnxsim', this.build_download_data(true), true, null, "Onnxsim Success");
    });

    const onnxOptimizer = this.document.getElementById('auto-optimizer-graph');
    onnxOptimizer.addEventListener('click', () => {
      if (window.DISPLAY_OM_MODEL) {
        this.show_alert_message('disabled', 'This button is disabled when displaying om model.');
        return;
      }
      this.take_effect_modify('/auto-optimizer', this.build_download_data(true), true, null, "Optimizer Success");
    });

    const extract = this.document.getElementById('extract-graph');
    extract.addEventListener('dblclick', () => {
      if (window.DISPLAY_OM_MODEL) {
        this.show_alert_message('disabled', 'This button is disabled when displaying om model.');
        return;
      }
      let start_nodes = this._view.modifier.getExtractStart();
      let end_nodes = this._view.modifier.getExtractEnd();
      if (!start_nodes || start_nodes.size == 0 || !end_nodes || end_nodes.size == 0) {
        this.show_message(
          'Select Extract Net Start And End',
          'Select the start node and end node for the subnet export',
          'warn'
        );
        return;
      }
      let download_data = this.build_download_data(true);
      download_data['extract_start'] = Array.from(start_nodes).join(',');
      download_data['extract_end'] = Array.from(end_nodes).join(',');
      download_data['session'] = this.session;
      this.take_effect_modify('/extract', download_data, false, (blob) => {
        this.export(this.upload_filename.replace('.onnx', '.extract.onnx'), blob);
        this.show_message('Success!', 'Model has been successfuly extracted', 'success');
        for (const start_name of start_nodes) {
          this._view.modifier.setExtractStart(start_name, false);
        }
        for (const end_name of end_nodes) {
          this._view.modifier.setExtractEnd(end_name, false);
        }
      });
    });

    this._menu_extract = new host.Dropdown(this, 'extract-graph', 'menu-extract-dropdown');
    this._menu_extract.add({
      label: 'Extract',
      accelerator: 'CmdOrCtrl+T',
      click: () => extract.dispatchEvent(new Event('dblclick')),
    });
    this._menu_extract.add({});
    this._menu_extract.add({
      label: (enable) =>
        enable ? this.document.getElementsByClassName('NodeExtractStart')[0].innerText : 'Extract Net Start',
      accelerator: 'CmdOrCtrl+S',
      enable: () => {
        return (
          this.document.getElementById('sidebar').style.display != 'none' &&
          this.document.getElementsByClassName('NodeExtractStart').length > 0
        );
      },
      click: () => this.document.getElementsByClassName('NodeExtractStart')[0].click(),
    });
    this._menu_extract.add({
      label: (enable) =>
        enable ? this.document.getElementsByClassName('NodeExtractEnd')[0].innerText : 'Extract Net End',
      accelerator: 'CmdOrCtrl+E',
      enable: () => {
        return (
          this.document.getElementById('sidebar').style.display != 'none' &&
          this.document.getElementsByClassName('NodeExtractEnd').length > 0
        );
      },
      click: () => this.document.getElementsByClassName('NodeExtractEnd')[0].click(),
    });

    extract.addEventListener('click', (e) => {
      if (window.DISPLAY_OM_MODEL) {
        this.show_alert_message('disabled', 'This button is disabled when displaying om model.');
        return;
      }
      let top = e.clientY;
      this.document.getElementById('menu-extract-dropdown').style.top = `${top}px`;
      this._menu_extract.toggle();
      e.preventDefault();
    });

    const addNodeButton = this.document.getElementById('add-node');
    addNodeButton.addEventListener('click', () => {
      if (window.DISPLAY_OM_MODEL) {
        this.show_alert_message('disabled', 'This button is disabled when displaying om model.');
        return;
      }
      let dialog = this.document.getElementById('addnode-dialog');
      this.show_confirm_dialog(dialog).then((is_not_cancel) => {
        if (!is_not_cancel) {
          return;
        }

        var addNodeDropDown = this.document.getElementById('add-node-dropdown');
        var selected_val = addNodeDropDown.options[addNodeDropDown.selectedIndex].value;
        var add_op_domain = selected_val.split(':')[0];
        var add_op_type = selected_val.split(':')[1];
        this._view.modifier.addNode(add_op_domain, add_op_type);
        this._view._updateGraph();
      });
    });

    function isJSONValid(jsonString) {
      try {
        jsonString = jsonString.trim();
        const jsonArray = JSON.parse(jsonString);
        if (!Array.isArray(jsonArray)) {
          return false;
        }
        for (const obj of jsonArray) {
          if (typeof obj !== 'object' || obj === null) {
            return false; // Each item must be a non-null object
          }
        }
        return true;
      } catch (e) {
        return false;
      }
    }

    function isValidInput(str) {
      var regex = /^[a-zA-Z0-9-_]+$/;
      return regex.test(str);
    }
    // 检查字符串是否为正数
    function isPositiveNumeric(str) {
      var num = parseFloat(str);
      return !isNaN(num) && isFinite(num) && num > 0;
    }

    function validateCustomOperatorForm() {
      if (!document.getElementById('customName').value.trim()) {
        window._host.show_message('Warn', 'Name field is required.', 'warn');
        return false;
      }
      if (!document.getElementById('customModule').value.trim()) {
        window._host.show_message('Warn', 'Module field is required.', 'warn');
        return false;
      }

      var nameValue = document.getElementById('customName').value.trim();
      var moduleValue = document.getElementById('customModule').value.trim();
      var supportLevelValue = document.getElementById('customSupportLevel').value.trim();
      var descriptionValue = document.getElementById('customDescription').value.trim();
      if (!isValidInput(nameValue) || !isValidInput(moduleValue)) {
        window._host.show_message('Warn', 'Name or Module field contains invalid characters.', 'warn');
        return false;
      }
      if (supportLevelValue && !isValidInput(supportLevelValue)) {
        window._host.show_message('Warn', 'Support Level field contains invalid characters.', 'warn');
        return false;
      }
      if (
        !validateStringLength(nameValue, 'Name') ||
        !validateStringLength(moduleValue, 'Module') ||
        !validateStringLength(supportLevelValue, 'Support Level') ||
        !validateStringLength(descriptionValue, 'Description')
      ) {
        return false; // 错误信息会在 validateStringLength 函数中处理
      }

      if (!document.getElementById('customVersion').value.trim()) {
        window._host.show_message('Warn', 'Version field is required.', 'warn');
        return false;
      }

      var versionValue = document.getElementById('customVersion').value;
      if (!isPositiveNumeric(versionValue)) {
        window._host.show_message('Warn', 'Version field must be a valid number.', 'warn');
        return false;
      }

      var customInputsValue = document.getElementById('customInputs').value;

      if (!customInputsValue.trim()) {
        window._host.show_message('Warn', 'Inputs field cannot be empty.', 'warn');
        return false;
      }

      if (!validateStringLength(customInputsValue, 'Inputs')) {
        return false;
      }

      if (!isJSONValid(document.getElementById('customInputs').value)) {
        window._host.show_message('Warn', 'Inputs field contains invalid JSON.', 'warn');
        return false;
      }

      var inputs = JSON.parse(document.getElementById('customInputs').value);
      if (!validateJSONContainsName(inputs)) {
        return false;
      }

      var customOutputsValue = document.getElementById('customOutputs').value;

      if (!customOutputsValue.trim()) {
        window._host.show_message('Warn', 'Outputs field cannot be empty.', 'warn');
        return false;
      }

      if (!validateStringLength(customOutputsValue, 'Outputs')) {
        return false;
      }

      if (!isJSONValid(document.getElementById('customOutputs').value)) {
        window._host.show_message('Warn', 'Outputs field contains invalid JSON.', 'warn');
        return false;
      }

      var outputs = JSON.parse(document.getElementById('customOutputs').value);
      if (!validateJSONContainsName(outputs)) {
        return false;
      }

      var customAttributes = document.getElementById('customAttributes').value;

      if (!validateStringLength(customAttributes, 'Attributes')) {
        return false;
      }

      if (
        document.getElementById('customAttributes').value &&
        !isJSONValid(document.getElementById('customAttributes').value)
      ) {
        window._host.show_message('Warn', 'Attributes field contains invalid JSON.', 'warn');
        return false;
      }

      var customTypeConstraints = document.getElementById('customTypeConstraints').value;

      if (!validateStringLength(customTypeConstraints, 'Type Constraints')) {
        return false;
      }

      if (
        document.getElementById('customTypeConstraints').value &&
        !isJSONValid(document.getElementById('customTypeConstraints').value)
      ) {
        window._host.show_message('Warn', 'Type Constraints field contains invalid JSON.', 'warn');
        return false;
      }

      return true;
    }

    function validateStringLength(stringValue, fieldName) {
      const MAX_STRING_LENGTH = 10000; // 最大长度，例如 10000 字符
      if (stringValue.length > MAX_STRING_LENGTH) {
        window._host.show_message('Warn', fieldName + ' field data is too long.', 'warn');
        return false;
      }
      return true;
    }

    function validateJSONContainsName(jsonArray) {
      for (var i = 0; i < jsonArray.length; i++) {
        if (!jsonArray[i].name) {
          window._host.show_message('Warn', 'Each item in JSON array must contain a "name" field.', 'warn');
          return false;
        }
      }
      return true;
    }

    document.getElementById('openCustomOperatorDialog').addEventListener('click', function () {
      document.getElementById('customOperatorDialog').showModal();
    });

    document.getElementById('customOperatorForm').addEventListener('submit', function (event) {
      event.preventDefault();
      if (validateCustomOperatorForm()) {
        submitCustomOperator();
      }
    });

    document.addEventListener('customOperatorAdded', function () {
      onnx.Metadata.reload(this.context)
        .then(() => {
          window.__view__.model.graphMetadata._metadata = onnx.Metadata._metadata;
          window.__view__.modifier.updateAddNodeDropDown();
        })
        .catch((error) => {
          window._host.show_message(
            'Error',
            'You are kindly to check the log and create an issue on https://gitee.com/ascend/ait.',
            'error'
          );
        });
    });

    // 函数：从服务器获取最新的算子列表
    function updateOperatorDropdown() {
      fetch('/get-operators', {
        method: 'POST',
        body: "{}",
      }).then((response) => response.json())
        .then((data) => {
          var addNodeDropdown = document.getElementById('add-node-dropdown');
          addNodeDropdown.innerHTML = ''; // 清空现有选项
          data.forEach((operator) => {
            var option = new Option(operator.name, operator.module + ':' + operator.name);
            addNodeDropdown.appendChild(option);
          });
        })
        .catch((error) =>
          window._host.show_message(
            'Error',
            'You are kindly to check the log and create an issue on https://gitee.com/ascend/ait.',
            'error'
          )
        );
    }

    function addNodeAutomatically(operatorName) {
      var addNodeDropDown = document.getElementById('add-node-dropdown');
      for (const node of window.__view__.model.supported_nodes) {
        var option = new Option(node[1], node[0] + ':' + node[1]);
        addNodeDropDown.appendChild(option);
      }

      for (var i = addNodeDropDown.options.length - 1; i > 0; i--) {
        if (addNodeDropDown.options[i].text === operatorName) {
          addNodeDropDown.selectedIndex = i;
          var selectedVal = addNodeDropDown.options[i].value;
          var addOpDomain = selectedVal.split(':')[0];
          var addOpType = selectedVal.split(':')[1];
          window.__view__.modifier.addNode(addOpDomain, addOpType);
          window.__view__._updateGraph();
          break;
        }
      }
    }

    function submitCustomOperator() {
      var customOperatorData = {
        name: document.getElementById('customName').value,
        module: document.getElementById('customModule').value,
        version: parseInt(document.getElementById('customVersion').value),
        support_level: document.getElementById('customSupportLevel').value,
        description: document.getElementById('customDescription').value,
        inputs: JSON.parse(document.getElementById('customInputs').value),
        outputs: JSON.parse(document.getElementById('customOutputs').value),
      };

      // 如果存在 Attributes，则添加到对象中
      if (document.getElementById('customAttributes').value) {
        customOperatorData.attributes = JSON.parse(document.getElementById('customAttributes').value);
      }

      // 如果存在 Type Constraints，则添加到对象中
      if (document.getElementById('customTypeConstraints').value) {
        customOperatorData.type_constraints = JSON.parse(document.getElementById('customTypeConstraints').value);
      }

      lastAddedOperatorName = customOperatorData.name;

      fetch('/add-custom-operator', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(customOperatorData),
      })
        .then((response) => response.json())
        .then((data) => {
          document.getElementById('customOperatorDialog').close();
          updateOperatorDropdown(); // 更新下拉菜单
          document.dispatchEvent(new CustomEvent('customOperatorAdded'));
          setTimeout(() => {
            window.__view__.modifier.updateAddNodeDropDown();
            addNodeAutomatically(customOperatorData.name);
          }, 500);
          window._host.updateCustomOperatorLists();
          window._host.show_message('Success!', 'Custom Operator has been successfuly Submit', 'success');
        })
        .catch((error) => {
          window._host.show_message(
            'Error',
            'Submit Error, You are kindly to check the log and create an issue on https://gitee.com/ascend/ait.',
            'error'
          );
        });
    }

    // 删除算子按钮事件监听器
    document.getElementById('deleteOperatorButton').addEventListener('click', function () {
      document.getElementById('deleteOperatorDialog').showModal();
    });

    // 确认删除算子按钮事件监听器
    document.getElementById('confirmDeleteOperatorButton').addEventListener('click', function () {
      const operatorName = document.getElementById('customName').value;
      const operatorModule = document.getElementById('customModule').value;
      const operatorVersion = parseInt(document.getElementById('customVersion').value, 10);

      if (['ai.onnx', 'com.microsoft', 'ai.onnx.preview.training', 'ai.onnx.ml'].includes(operatorModule)) {
        window._host.show_message('Warn', 'Cannot delete core operator.', 'warn');
        return;
      }

      fetch('/delete-custom-operator', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: operatorName, module: operatorModule, version: operatorVersion }),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error('Delete Error');
          }
          return response.json();
        })
        .then((data) => {
          updateOperatorDropdown();
          setTimeout(() => {
            window.__view__.modifier.updateAddNodeDropDown();
            window.__view__.model.graphMetadata._metadata = onnx.Metadata._metadata;
          }, 500);
          var addNodeDropdown = document.getElementById('add-node-dropdown');
          for (const node of window.__view__.model.supported_nodes) {
            var option = new Option(node[1], node[0] + ':' + node[1]);
            addNodeDropdown.appendChild(option);
          }
          window._host.updateCustomOperatorLists();
          updateOperatorDropdown();
          window._host.show_message('Success!', 'Operator deleted successfully', 'success');
          document.getElementById('deleteOperatorDialog').close();
        })
        .catch((error) => {
          window._host.show_message('Error', 'Delete Error or Not Found Operator', 'error');
        });
    });

    this.document.getElementById('version').innerText = this.version;

    if (this._meta.file) {
      const url = this._meta.file[0];
      if (this._view.accept(url)) {
        this._openModel(this._url(url), null);
        return;
      }
    }

    const openFileButton = this.document.getElementById('open-file-button');
    const openFileButtonLogo = this.document.getElementById('open-file-button-logo');
    const openFileDialog = this.document.getElementById('open-file-dialog');
    if (openFileButtonLogo && openFileDialog) {
      openFileButton.addEventListener('click', () => {
        openFileDialog.value = '';
        openFileDialog.click();
      });
      openFileButtonLogo.addEventListener('click', () => {
        openFileDialog.value = '';
        openFileDialog.click();
      });
      openFileDialog.addEventListener('change', (e) => {
        if (e.target && e.target.files && e.target.files.length > 0) {
          const files = Array.from(e.target.files);
          const file = files.find((file) => this._view.accept(file.name));
          this.upload_filename = file.name;
          this.upload_filepath = file.path;
          var form = new FormData();
          form.append('file', file);
          this._ori_model_file = file;
          form.append('session', this.session);

          showWaitPage(true)
          fetch('/open_model', {
            method: 'POST',
            body: form,
          })
            .then((response) => {
              this.check_res_status(response.status);
              return response.text();
            })
            .then(function (text) {
              console.log('POST response: ');
              // Should be 'OK' if everything was successful
              console.log(text);
              showWaitPage(false)
            });

          if (file) {
            this._open(file, files, true);
          }
        }
      });
    }

    const githubButton = this.document.getElementById('github-button');
    if (githubButton) {
      githubButton.style.opacity = 1;
      githubButton.addEventListener('click', () => {
        this.openURL('https://gitee.com/ascend/ait/tree/master/onnx-modifier');
      });
    }
    this.document.addEventListener('dragover', (e) => {
      e.preventDefault();
    });
    this.document.addEventListener('drop', (e) => {
      e.preventDefault();
    });
    this.document.body.addEventListener('drop', (e) => {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const files = Array.from(e.dataTransfer.files);
        const file = files.find((file) => this._view.accept(file.name));
        this.upload_filename = file.name;
        this.upload_filepath = file.path;
        var form = new FormData();
        form.append('file', file);
        this._ori_model_file = file;
        form.append('session', this.session);

        showWaitPage(true)
        fetch('/open_model', {
          method: 'POST',
          body: form,
        })
          .then((response) => {
            this.check_res_status(response.status);
            return response.text();
          })
          .then(function (text) {
            console.log('POST response: ');
            // Should be 'OK' if everything was successful
            console.log(text);
          });
        if (file) {
          this._open(file, files, true);
        }
      }
    });
    this.init_dragable();

    this._view.show('welcome');
  }

  init_dragable() {
    let dragable = this.document.getElementById('dragable');
    let sidebar = this.document.getElementById('sidebar');
    let oriX = -1;
    let minWidth = sidebar.getBoundingClientRect().width;
    let oriWidth = sidebar.getBoundingClientRect().width;
    let dragableMouseMove = function (e) {
      if (minWidth > oriWidth + oriX - e.clientX) {
        return;
      }
      sidebar.style.width = `${oriWidth + oriX - e.clientX}px`;
    };
    let dragableMouseUp = function (e) {
      sidebar.style.transition = null;
      document.removeEventListener('mousemove', dragableMouseMove);
      document.removeEventListener('mouseup', dragableMouseUp);
    };
    let dragableMouseDown = function (e) {
      oriX = e.clientX;
      oriWidth = sidebar.getBoundingClientRect().width;
      if (minWidth == 0) {
        minWidth = oriWidth;
      }
      sidebar.style.transition = `none`;
      document.addEventListener('mousemove', dragableMouseMove);
      document.addEventListener('mouseup', dragableMouseUp);
    };
    dragable.addEventListener('mousedown', dragableMouseDown);
  }

  get_default_input_shape(input_name) {
    let default_shape = '';
    for (var input_info of this._view.modifier.graph.inputs) {
      if (input_name == input_info.name) {
        if (
          input_info.arguments &&
          input_info.arguments.length > 0 &&
          input_info.arguments[0].type &&
          input_info.arguments[0].type.shape &&
          input_info.arguments[0].type.shape.dimensions &&
          input_info.arguments[0].type.shape.dimensions.length > 0
        ) {
          let dims = input_info.arguments[0].type.shape.dimensions;
          default_shape = dims.map((dim) => (dim ? dim.toString() : '?')).join(',');
        }
        break;
      }
    }
    return default_shape;
  }

  dimStr2dimArray(dim_str) {
    let dims = [];
    let has_error = false;
    let input_dims = dim_str.split(',');
    for (const dim_str of input_dims) {
      let dim = dim_str.trim();
      if (dim.match('^-?[1-9][0-9]{0,10}$')) {
        dims.push(parseInt(dim));
      } else if (dim.match('^[a-zA-Z\\-_\\\\/\\.0-9]{1,64}$')) {
        dims.push(dim);
      } else {
        has_error = true;
      }
    }
    return [dims, has_error];
  }

  change_batch_size(batch_value) {
    for (var input_info of this._view.modifier.graph.inputs) {
      if (
        input_info.arguments &&
        input_info.arguments.length > 0 &&
        input_info.arguments[0].type &&
        input_info.arguments[0].type.shape &&
        input_info.arguments[0].type.shape.dimensions &&
        input_info.arguments[0].type.shape.dimensions.length > 0
      ) {
        let dims = input_info.arguments[0].type.shape.dimensions;
        if (dims.length > 0) {
          let [ori_value, pp] = this.dimStr2dimArray(
            dims
              .concat()
              .map((dim) => (dim ? dim.toString() : '?'))
              .join(',')
          );
          dims[0] = batch_value.toString();
          let dim_str = dims.map((dim) => (dim ? dim.toString() : '?')).join(',');
          let [input_dims, has_error] = this.dimStr2dimArray(dim_str);
          this._view.modifier.changeInputSize(input_info.name, input_dims, ori_value);
        }
      }
    }
  }

  init_input_shape_change_event() {
    let input_change = this.document.getElementById('change-input-shape-input');
    input_change.addEventListener('input', (e) => {
      let value = e.target.value.trim();

      let [dims, has_error] = this.dimStr2dimArray(value);

      if (has_error) {
        input_change.classList.add('input_error');
        input_change.has_error = true;
        this.document.getElementById('change-input-shape-input-ok').disabled = 'disabled';
      } else {
        input_change.classList.remove('input_error');
        input_change.has_error = false;
        input_change.dims = dims;
        this.document.getElementById('change-input-shape-input-ok').disabled = '';
      }
    });
  }
  init_batch_size_change_event() {
    let input_change = this.document.getElementById('fixed-batch-size-input');
    input_change.addEventListener('input', (e) => {
      let value = e.target.value.trim();
      if (!value.match('^-?[1-9][0-9]{0,10}$')) {
        input_change.classList.add('input_error');
        input_change.has_error = true;
        this.document.getElementById('fixed-batch-size-input-ok').disabled = 'disabled';
      } else {
        input_change.classList.remove('input_error');
        input_change.has_error = false;
        this.document.getElementById('fixed-batch-size-input-ok').disabled = '';
      }
    });
  }

  bolb2text(blob) {
    return new Promise((resolve) => {
      let reader = new FileReader();
      reader.readAsText(blob, 'utf-8');
      reader.onload = () => {
        resolve(reader.result);
      };
    });
  }

  take_effect_modify(path, data_body, record_modify, callback, successMsg) {
    let isSuccessed = false;
    showWaitPage(true);
    
    return fetch(path, {
      // Declare what type of data we're sending
      headers: {
        'Content-Type': 'application/json',
      },
      // Specify the method
      method: 'POST',
      body: typeof data_body == 'string' ? data_body : JSON.stringify(data_body),
    })
      .then((response) => {
        if (this.check_res_status(response.status)) {
          return;
        } else if (response.status == 299) {
          return response.text().then((text) => {
            this.show_message('Nothing happens!', text, 'success');
          });
        } else if (!response.ok) {
          return response.text().then((text) => {
            this.show_message(
              'Error happens!',
              `You are kindly to check the log and create an issue on https://gitee.com/ascend/ait\n${text}`,
              'error'
            );
          });
        }

        isSuccessed = true;
        if (record_modify) {
          this._modify_info.push({ path, data_body });
        }
        return response.blob();
      })
      .then((blob) => {
        if (!blob) {
          showWaitPage(false)
          return;
        }
        if (callback) {
          showWaitPage(false)
          return callback(blob);
        }

        let file = new File([blob], this.upload_filename);
        file.filepath = blob.filepath ? blob.filepath : this.upload_filepath;
        return this.openFile(file);
      })
      .then(() => {
        if (successMsg && isSuccessed) {
          this.show_message('Success', successMsg, 'success');
        }

        let body = JSON.stringify({ session: this.session });
        fetch('/get_output_message', {
          headers: {
            'Content-Type': 'application/json',
          },
          method: 'POST',
          body,
        }).then((response) => {
          response.text().then((text) => {
            if (text) {
              this.show_message('Messages', text, 'info');
            }
          });
        });
      });
  }

  check_res_status(status) {
    if (status == 598) {
      this.show_message(
        'Error',
        'Server Error, you can save modify info to json and reopen this page, and then raise an issue.',
        'error'
      );
      return true;
    }
    return false;
  }

  openFile(file) {
    let files = [file];

    let form = new FormData();
    form.append('file', file);
    form.append('session', this.session);

    fetch('/open_model', {
      method: 'POST',
      body: form,
    })
      .then((response) => {
        this.check_res_status(response.status);
        return response.text();
      })
      .then(function (text) {
        console.log('POST response: ');
        // Should be 'OK' if everything was successful
        console.log(text);
      }).then(()=>{
        if (file) {
          this._activate_model_file = file;
          return this._open(file, files);
        }
      });
  }

  build_download_data(return_modified_file, shape_inf) {
    return {
      node_states: this.mapToObjectRec(this._view.modifier.name2NodeStates),
      node_renamed_io: this.mapToObjectRec(this._view.modifier.renameMap),
      node_changed_attr: this.mapToObjectRec(this._view.modifier.changedAttributes),
      added_node_info: this.mapToObjectRec(
        this.parseAddedLightNodeInfo2Map(this._view.modifier.addedNode, this._view.modifier.initializerEditInfo)
      ),
      added_outputs: this.arrayToObject(
        this.process_added_outputs(
          this._view.modifier.addedOutputs,
          this._view.modifier.renameMap,
          this._view.modifier.name2NodeStates
        )
      ),
      added_inputs: this.arrayToObject(
        this.process_added_inputs(
          this._view.modifier.addedInputs,
          this._view.modifier.renameMap,
          this._view.modifier.name2NodeStates
        )
      ),
      rebatch_info: this.mapToObjectRec(this._view.modifier.reBatchInfo),
      changed_initializer: this.mapToObjectRec(this._view.modifier.initializerEditInfo),
      postprocess_args: {
        shapeInf: this._view.modifier.downloadWithShapeInf,
        cleanUp: this._view.modifier.downloadWithCleanUp,
      },
      model_properties: this.mapToObjectRec(this._view.modifier.modelProperties),
      input_size_info: this.mapToObjectRec(this._view.modifier.inputSizeInfo),
      return_modified_file: Boolean(return_modified_file),
      session: this.session,
    };
  }

  environment(name) {
    return this._environment.get(name);
  }

  show_message(title, message, level) {
    let box = document.createElement('div');
    box.classList.add('message-box', `${level}-message-box`, 'message-box-disappear');
    let progressLine = document.createElement('div');
    progressLine.classList.add('message-box-progress', `message-box-progress-${level}`);

    let boxTitle = document.createElement('b');
    boxTitle.innerText = title;
    let boxClose = document.createElement('span');
    boxClose.innerText = '[ X ]';
    boxClose.style.float = 'right';
    let boxText = document.createElement('p');
    boxText.classList.add('text');
    boxText.innerText = message;
    box.append(progressLine, boxTitle, boxClose, boxText);
    document.getElementById('show-message-info').append(box);

    // event
    let remove_function = () => {
      box.remove();
    };
    boxClose.addEventListener('click', remove_function);
    box.addEventListener('animationend', remove_function);
  }

  show_alert_message(title, message) {
    let alert_element = document.getElementById('show-message-alert');
    alert_element.getElementsByTagName('h1')[0].innerText = title;
    alert_element.getElementsByClassName('text')[0].innerText = message;
    alert_element.showModal();
  }

  show_confirm_message(title, message, buttons) {
    let confirm_element = document.getElementById('show-message-confirm');
    confirm_element.getElementsByTagName('h1')[0].innerText = title;
    confirm_element.getElementsByClassName('text')[0].innerText = message;
    if (!buttons) {
      buttons = { Cancel: '', OK: 'OK' };
    }
    let footer = confirm_element.getElementsByClassName('footer')[0];
    while (footer.lastChild) {
      footer.removeChild(footer.lastChild);
    }
    for (const btnText in buttons) {
      const value = buttons[btnText];
      let buttonElem = this.document.createElement('button');
      buttonElem.innerText = btnText;
      buttonElem.dataset.value = value;
      footer.appendChild(buttonElem);
    }

    return this.show_confirm_dialog(confirm_element);
  }

  show_confirm_dialog(dialogElem) {
    return new Promise((resolve) => {
      let btns = dialogElem.getElementsByTagName('button');
      let listener = [];
      let remove_listener = () => {
        for (const [btn, cancel_listener] of listener) {
          btn.removeEventListener('click', cancel_listener);
        }
      };

      for (const btn of btns) {
        listener.push([
          btn,
          btn.addEventListener('click', () => {
            dialogElem.close();
            remove_listener();
            resolve(btn.dataset.value);
          }),
        ]);
      }
      dialogElem.showModal();
    });
  }

  error(message, detail) {
    this.show_alert_message(message, detail);
  }

  confirm(message, detail, buttons) {
    return this.show_confirm_message(message, detail, buttons);
  }

  require(id) {
    const url = this._url('./' + id + '.js');
    this.window.__modules__ = this.window.__modules__ || {};
    if (this.window.__modules__[url]) {
      return Promise.resolve(this.window.__exports__[url]);
    }
    return new Promise((resolve, reject) => {
      this.window.module = { exports: {} };
      const script = document.createElement('script');
      script.setAttribute('id', id);
      script.setAttribute('type', 'text/javascript');
      script.setAttribute('src', url);
      script.onload = (e) => {
        if (this.window.module && this.window.module.exports) {
          const exports = this.window.module.exports;
          delete this.window.module;
          this.window.__modules__[id] = exports;
          resolve(exports);
        } else {
          reject(new Error("The script '" + e.target.src + "' has no exports."));
        }
      };
      script.onerror = (e) => {
        delete this.window.module;
        reject(new Error("The script '" + e.target.src + "' failed to load."));
      };
      this.document.head.appendChild(script);
    });
  }

  save(name, extension, defaultPath, callback) {
    callback(defaultPath + '.' + extension);
  }

  export(file, blob) {
    const element = this.document.createElement('a');
    element.download = file;
    element.href = URL.createObjectURL(blob);
    this.document.body.appendChild(element);
    element.click();
    this.document.body.removeChild(element);
  }

  request(file, encoding, base) {
    const url = base ? base + '/' + file : this._url(file);
    return this._request(url, null, encoding);
  }

  openURL(url) {
    this.window.open(url);
  }

  exception(error, fatal) {
    if (this._telemetry && this.window.ga && error.telemetry !== false) {
      const description = [];
      description.push(
        (error && error.name ? error.name + ': ' : '') + (error && error.message ? error.message : '(null)')
      );
      if (error.stack) {
        const match = error.stack.match(/\n {4}at (.{1,100})\((.{1,100})\)/);
        if (match) {
          description.push(match[1] + '(' + match[2].split('/').pop() + ')');
        } else {
          description.push(error.stack.split('\n').shift());
        }
      }
      this.window.ga('send', 'exception', {
        exDescription: description.join(' @ '),
        exFatal: fatal,
        appName: this.type,
        appVersion: this.version,
      });
    }
  }

  screen(name) {
    if (this._telemetry && this.window.ga) {
      this.window.ga('send', 'screenview', {
        screenName: name,
        appName: this.type,
        appVersion: this.version,
      });
    }
  }

  event(category, action, label, value) {
    if (this._telemetry && this.window.ga) {
      this.window.ga('send', 'event', {
        eventCategory: category,
        eventAction: action,
        eventLabel: label,
        eventValue: value,
        appName: this.type,
        appVersion: this.version,
      });
    }
  }

  _request(url, headers, encoding, timeout) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      if (!encoding) {
        request.responseType = 'arraybuffer';
      }
      if (timeout) {
        request.timeout = timeout;
      }
      const error = (status) => {
        const err = new Error('The web request failed with status code ' + status + " at '" + url + "'.");
        err.type = 'error';
        err.url = url;
        return err;
      };
      request.onload = () => {
        if (request.status == 200) {
          if (request.responseType == 'arraybuffer') {
            resolve(new host.BrowserHost.BinaryStream(new Uint8Array(request.response)));
          } else {
            resolve(request.responseText);
          }
        } else {
          reject(error(request.status));
        }
      };
      request.onerror = (e) => {
        const err = error(request.status);
        err.type = e.type;
        reject(err);
      };
      request.ontimeout = () => {
        request.abort();
        const err = new Error("The web request timed out in '" + url + "'.");
        err.type = 'timeout';
        err.url = url;
        reject(err);
      };
      request.open('GET', url, true);
      if (headers) {
        for (const name of Object.keys(headers)) {
          request.setRequestHeader(name, headers[name]);
        }
      }
      request.send();
    });
  }

  _url(file) {
    let url = file;
    if (this.window && this.window.location && this.window.location.href) {
      let location = this.window.location.href.split('?').shift();
      if (location.endsWith('.html')) {
        location = location.split('/').slice(0, -1).join('/');
      }
      if (location.endsWith('/')) {
        location = location.slice(0, -1);
      }
      url = location + '/' + (file.startsWith('/') ? file.substring(1) : file);
    }
    return url;
  }

  _openModel(url, identifier) {
    url = url + (/\?/.test(url) ? '&' : '?') + 'cb=' + new Date().getTime();
    this._view.show('welcome spinner');
    this._request(url)
      .then((buffer) => {
        const context = new host.BrowserHost.BrowserContext(this, url, identifier, buffer);
        this._view
          .open(context)
          .then(() => {
            this.document.title = identifier || context.identifier;
          })
          .catch((err) => {
            if (err) {
              this._view.error(err, null, 'welcome');
            }
          });
      })
      .catch((err) => {
        this.error('Model load request failed.', err.message);
        this._view.show('welcome');
      });
  }

  _open(file, files, cleanSubGraph) {
    if (cleanSubGraph) {
      this._view.clearSubGraph();
    }
    this._view.show('welcome spinner');
    const context = new host.BrowserHost.BrowserFileContext(this, file, files);
    context
      .open()
      .then(() => {
        return this._view.open(context).then((model) => {
          this._view.show(null);
          this._updateButtons();
          this.document.title = files[0].name;
          return model;
        });
      })
      .catch((error) => {
        this._view.error(error, null, null);
      });
  }
  updateCustomOperatorLists() {
    fetch('/get-operators', {
      method: 'POST',
      body: "{}",
    }).then((response) => response.json())
      .then((data) => {
        const nameList = document.getElementById('customNameList');
        const moduleList = document.getElementById('customModuleList');
        nameList.innerHTML = ''; // 清空现有选项
        moduleList.innerHTML = ''; // 清空现有选项

        data.forEach((operator) => {
          if (!['ai.onnx', 'com.microsoft', 'ai.onnx.preview.training', 'ai.onnx.ml'].includes(operator.module)) {
            const nameOption = document.createElement('option');
            nameOption.value = operator.name;
            nameList.appendChild(nameOption);

            const moduleOption = document.createElement('option');
            moduleOption.value = operator.module;
            moduleList.appendChild(moduleOption);
          }
        });
      })
      .catch((error) => console.error('Error:', error));
  }
  _updateButtons() {
    let idList = [
      'download-graph',
      'reset-graph',
      'extract-graph',
      'add-node',
      'onnxsim-graph',
      'auto-optimizer-graph',
      'modify-export',
      'modify-import',
    ];
    if (window.DISPLAY_OM_MODEL) {
      for (var id of idList) {
        let tmpButton = this.document.getElementById(id);
        tmpButton.style.display = 'none';
      }
    } else {
      for (var id of idList) {
        let tmpButton = this.document.getElementById(id);
        tmpButton.style.display = 'block';
      }
    }
  }

  _setCookie(name, value, days) {
    const date = new Date();
    date.setTime(date.getTime() + (typeof days !== 'number' ? 365 : days) * 24 * 60 * 60 * 1000);
    document.cookie = name + '=' + value + ';path=/;expires=' + date.toUTCString();
  }

  _getCookie(name) {
    const cookie = '; ' + document.cookie;
    const parts = cookie.split('; ' + name + '=');
    return parts.length < 2 ? undefined : parts.pop().split(';').shift();
  }

  _about() {
    document.getElementById('show-about').showModal();
  }

  _strMapToObj(strMap) {
    let obj = Object.create(null);
    for (let [k, v] of strMap) {
      obj[k] = v;
    }
    return obj;
  }

  // {key1:val1, key2:val2, ...} => Json
  _mapToJson(map) {
    return JSON.stringify(this._strMapToObj(map));
  }

  mapToObjectRec(m) {
    let lo = {};
    for (let [k, v] of m) {
      if (v instanceof Map) {
        lo[k] = this.mapToObjectRec(v);
      } else {
        lo[k] = v;
      }
    }
    return lo;
  }

  // this function does 2 things:
  // 1. rename the addedOutputs with their new names using renameMap. Because addedOutputs are stored in lists,
  //    it may be not easy to rename them while editing. (Of course there may be a better way to do this)
  // 2. filter out the custom output which is added, but deleted later
  process_added_outputs(addedOutputs, renameMap, modelNodeName2State) {
    var processed = [];
    for (var out of addedOutputs) {
      if (modelNodeName2State.get('out_' + out) == 'Exist') {
        processed.push(out);
      }
    }
    for (let i = 0; i < processed.length; ++i) {
      if (renameMap.get('out_' + processed[i])) {
        processed[i] = renameMap.get('out_' + processed[i]).get(processed[i]);
      }
    }
    return processed;
  }

  process_added_inputs(addedInputs, renameMap, modelNodeName2State) {
    var processed = [];
    for (var in_info of addedInputs) {
      if (modelNodeName2State.get(in_info) == 'Exist') {
        processed.push(in_info);
      }
    }
    for (let i = 0; i < processed.length; ++i) {
      if (renameMap.get(processed[i])) {
        processed[i] = renameMap.get(processed[i]).get(processed[i]);
      }
    }
    return processed;
  }

  arrayToObject(arr) {
    var rv = {};
    for (var i = 0; i < arr.length; ++i) if (arr[i] !== undefined) rv[i] = arr[i];
    return rv;
  }

  // convert view.LightNodeInfo to Map object for easier transmission to Python backend
  parseAddedLightNodeInfo2Map(nodes_info, initializer_info) {
    var res_map = new Map();
    for (const [modelNodeName, node_info] of nodes_info) {
      var node_info_map = new Map();
      node_info_map.set('properties', node_info.properties);
      node_info_map.set('attributes', node_info.attributes);

      // skip the input and output which is optional and has no initializer value
      var inputs = new Map();
      for (var [input_name, arg_list] of node_info.inputs) {
        var filtered_arg_list = [];
        for (var arg of arg_list) {
          var arg_name = arg[0],
            arg_optional = arg[1];
          if (arg_optional) {
            if (!initializer_info.get(arg_name) || initializer_info.get(arg_name) == '') {
              continue;
            }
          }
          filtered_arg_list.push(arg_name);
        }
        if (filtered_arg_list.length > 0) {
          inputs.set(input_name, filtered_arg_list);
        }
      }
      node_info_map.set('inputs', inputs);

      var outputs = new Map();
      for (var [output_name, arg_list] of node_info.outputs) {
        var filtered_arg_list = [];
        for (var arg of arg_list) {
          var arg_name = arg[0],
            arg_optional = arg[1];
          if (arg_optional) {
            if (!initializer_info.get(arg_name) || initializer_info.get(arg_name) == '') {
              continue;
            }
          }
          filtered_arg_list.push(arg_name);
        }
        if (filtered_arg_list.length > 0) {
          outputs.set(output_name, filtered_arg_list);
        }
      }
      node_info_map.set('outputs', outputs);

      res_map.set(modelNodeName, node_info_map);
    }

    return res_map;
  }

  // rename the initializer if its corresponding argument name is changed (for primitive nodes)
  process_initializer(initializer_info, rename_map) {
    for (const [node_name, rename_pair] of rename_map) {
      for (const [arg_orig_name, arg_new_name] of rename_pair) {
        if (initializer_info.has(arg_orig_name)) {
          var init_val = initializer_info.get(arg_orig_name);
          initializer_info.set(arg_new_name, init_val);
          initializer_info.delete(arg_orig_name);
        }
      }
    }
  }
};

host.Dropdown = class {
  constructor(host, button, dropdown) {
    this._host = host;
    this._dropdown = this._host.document.getElementById(dropdown);
    this._button = this._host.document.getElementById(button);
    this._items = [];
    this._apple = /(Mac|iPhone|iPod|iPad)/i.test(navigator.platform);
    this._acceleratorMap = {};
    this._host.window.addEventListener('keydown', (e) => {
      let code = e.keyCode;
      code |= (e.ctrlKey && !this._apple) || (e.metaKey && this._apple) ? 0x0400 : 0;
      code |= e.altKey ? 0x0200 : 0;
      code |= e.shiftKey ? 0x0100 : 0;
      if (code == 0x001b) {
        // Escape
        this.close();
        return;
      }
      const item = this._acceleratorMap[code.toString()];
      if (item) {
        item.click();
        e.preventDefault();
      }
    });
    this._host.document.body.addEventListener('click', (e) => {
      if (!this._button.contains(e.target)) {
        this.close();
      }
    });
  }

  add(item) {
    const accelerator = item.accelerator;
    if (accelerator) {
      let cmdOrCtrl = false;
      let alt = false;
      let shift = false;
      let key = '';
      for (const part of item.accelerator.split('+')) {
        switch (part) {
          case 'CmdOrCtrl':
            cmdOrCtrl = true;
            break;
          case 'Alt':
            alt = true;
            break;
          case 'Shift':
            shift = true;
            break;
          default:
            key = part;
            break;
        }
      }
      if (key !== '') {
        item.accelerator = {};
        item.accelerator.text = '';
        if (this._apple) {
          item.accelerator.text += alt ? '&#x2325;' : '';
          item.accelerator.text += shift ? '&#x21e7;' : '';
          item.accelerator.text += cmdOrCtrl ? '&#x2318;' : '';
          const keyTable = { Enter: '&#x23ce;', Up: '&#x2191;', Down: '&#x2193;', Backspace: '&#x232B;' };
          item.accelerator.text += keyTable[key] ? keyTable[key] : key;
        } else {
          const list = [];
          if (cmdOrCtrl) {
            list.push('Ctrl');
          }
          if (alt) {
            list.push('Alt');
          }
          if (shift) {
            list.push('Shift');
          }
          list.push(key);
          item.accelerator.text = list.join('+');
        }
        let code = 0;
        switch (key) {
          case 'Backspace':
            code = 0x08;
            break;
          case 'Enter':
            code = 0x0d;
            break;
          case 'Up':
            code = 0x26;
            break;
          case 'Down':
            code = 0x28;
            break;
          default:
            code = key.charCodeAt(0);
            break;
        }
        code |= cmdOrCtrl ? 0x0400 : 0;
        code |= alt ? 0x0200 : 0;
        code |= shift ? 0x0100 : 0;
        this._acceleratorMap[code.toString()] = item;
      }
    }
    this._items.push(item);
  }

  toggle() {
    if (this._dropdown.style.display === 'block') {
      this.close();
      return;
    }

    while (this._dropdown.lastChild) {
      this._dropdown.removeChild(this._dropdown.lastChild);
    }

    for (const item of this._items) {
      if (Object.keys(item).length > 0) {
        let enable = true;
        if (item.enable != undefined) {
          if (typeof item.enable == 'function') {
            enable = item.enable(item);
          } else {
            enable = item.enable;
          }
        }

        const button = this._host.document.createElement('button');
        button.innerText = typeof item.label == 'function' ? item.label(enable) : item.label;
        button.addEventListener('click', () => {
          this.close();
          setTimeout(() => {
            item.click();
          }, 10);
        });
        if (!enable) {
          button.disabled = 'disabled';
          button.style.cursor = 'not-allowed';
        }
        this._dropdown.appendChild(button);
        if (item.accelerator && enable) {
          const accelerator = this._host.document.createElement('span');
          accelerator.style.float = 'right';
          accelerator.innerHTML = item.accelerator.text;
          button.appendChild(accelerator);
        }
      } else {
        const separator = this._host.document.createElement('div');
        separator.setAttribute('class', 'separator');
        this._dropdown.appendChild(separator);
      }
    }

    this._dropdown.style.display = 'block';
  }

  close() {
    this._dropdown.style.display = 'none';
  }
};

host.BrowserHost.BinaryStream = class {
  constructor(buffer) {
    this._buffer = buffer;
    this._length = buffer.length;
    this._position = 0;
  }

  get position() {
    return this._position;
  }

  get length() {
    return this._length;
  }

  stream(length) {
    const buffer = this.read(length);
    return new host.BrowserHost.BinaryStream(buffer.slice(0));
  }

  seek(position) {
    this._position = position >= 0 ? position : this._length + position;
  }

  skip(offset) {
    this._position += offset;
  }

  peek(length) {
    if (this._position === 0 && length === undefined) {
      return this._buffer;
    }
    const position = this._position;
    this.skip(length !== undefined ? length : this._length - this._position);
    const end = this._position;
    this.seek(position);
    return this._buffer.subarray(position, end);
  }

  read(length) {
    if (this._position === 0 && length === undefined) {
      this._position = this._length;
      return this._buffer;
    }
    const position = this._position;
    this.skip(length !== undefined ? length : this._length - this._position);
    return this._buffer.subarray(position, this._position);
  }

  byte() {
    const position = this._position;
    this.skip(1);
    return this._buffer[position];
  }
};

host.BrowserHost.BrowserFileContext = class {
  constructor(host, file, blobs) {
    this._host = host;
    this._file = file;
    this._blobs = {};
    for (const blob of blobs) {
      this._blobs[blob.name] = blob;
    }
  }

  get identifier() {
    return this._file.name;
  }

  get stream() {
    return this._stream;
  }

  request(file, encoding, base) {
    if (base !== undefined) {
      return this._host.request(file, encoding, base);
    }
    const blob = this._blobs[file];
    if (!blob) {
      return Promise.reject(new Error("File not found '" + file + "'."));
    }
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(encoding ? e.target.result : new host.BrowserHost.BinaryStream(new Uint8Array(e.target.result)));
      };
      reader.onerror = (e) => {
        e = e || this.window.event;
        let message = '';
        const error = e.target.error;
        switch (error.code) {
          case error.NOT_FOUND_ERR:
            message = "File not found '" + file + "'.";
            break;
          case error.NOT_READABLE_ERR:
            message = "File not readable '" + file + "'.";
            break;
          case error.SECURITY_ERR:
            message = "File access denied '" + file + "'.";
            break;
          default:
            message = error.message ? error.message : "File read '" + error.code.toString() + "' error '" + file + "'.";
            break;
        }
        reject(new Error(message));
      };
      if (encoding === 'utf-8') {
        reader.readAsText(blob, encoding);
      } else {
        reader.readAsArrayBuffer(blob);
      }
    });
  }

  require(id) {
    return this._host.require(id);
  }

  exception(error, fatal) {
    this._host.exception(error, fatal);
  }

  open() {
    return this.request(this._file.name, null).then((stream) => {
      this._stream = stream;
    });
  }
};

host.BrowserHost.BrowserContext = class {
  constructor(host, url, identifier, stream) {
    this._host = host;
    this._stream = stream;
    if (identifier) {
      this._identifier = identifier;
      this._base = url;
      if (this._base.endsWith('/')) {
        this._base.substring(0, this._base.length - 1);
      }
    } else {
      const parts = url.split('?')[0].split('/');
      this._identifier = parts.pop();
      this._base = parts.join('/');
    }
  }

  get identifier() {
    return this._identifier;
  }

  get stream() {
    return this._stream;
  }

  request(file, encoding, base) {
    return this._host.request(file, encoding, base === undefined ? this._base : base);
  }

  require(id) {
    return this._host.require(id);
  }

  exception(error, fatal) {
    this._host.exception(error, fatal);
  }
};

if (typeof TextDecoder === 'undefined') {
  TextDecoder = function TextDecoder(encoding) {
    this._encoding = encoding;
  };
  TextDecoder.prototype.decode = function decode(buffer) {
    let result = '';
    const length = buffer.length;
    let i = 0;
    switch (this._encoding) {
      case 'utf-8':
        while (i < length) {
          const c = buffer[i++];
          switch (c >> 4) {
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7: {
              result += String.fromCharCode(c);
              break;
            }
            case 12:
            case 13: {
              const c2 = buffer[i++];
              result += String.fromCharCode(((c & 0x1f) << 6) | (c2 & 0x3f));
              break;
            }
            case 14: {
              const c2 = buffer[i++];
              const c3 = buffer[i++];
              result += String.fromCharCode(((c & 0x0f) << 12) | ((c2 & 0x3f) << 6) | ((c3 & 0x3f) << 0));
              break;
            }
            case 15: {
              const c2 = buffer[i++];
              const c3 = buffer[i++];
              const c4 = buffer[i++];
              result += String.fromCodePoint(
                ((c & 0x07) << 18) | ((c2 & 0x3f) << 12) | ((c3 & 0x3f) << 6) | (c4 & 0x3f)
              );
            }
          }
        }
        break;
      case 'ascii':
        while (i < length) {
          result += String.fromCharCode(buffer[i++]);
        }
        break;
    }
    return result;
  };
}

if (typeof TextEncoder === 'undefined') {
  TextEncoder = function TextEncoder() {};
  TextEncoder.prototype.encode = function encode(str) {
    'use strict';
    const length = str.length;
    let resPos = -1;
    const resArr = typeof Uint8Array === 'undefined' ? new Array(length * 2) : new Uint8Array(length * 3);
    for (let point = 0, nextcode = 0, i = 0; i !== length; ) {
      point = str.charCodeAt(i);
      i += 1;
      if (point >= 0xd800 && point <= 0xdbff) {
        if (i === length) {
          resArr[(resPos += 1)] = 0xef;
          resArr[(resPos += 1)] = 0xbf;
          resArr[(resPos += 1)] = 0xbd;
          break;
        }
        nextcode = str.charCodeAt(i);
        if (nextcode >= 0xdc00 && nextcode <= 0xdfff) {
          point = (point - 0xd800) * 0x400 + nextcode - 0xdc00 + 0x10000;
          i += 1;
          if (point > 0xffff) {
            resArr[(resPos += 1)] = (0x1e << 3) | (point >>> 18);
            resArr[(resPos += 1)] = (0x2 << 6) | ((point >>> 12) & 0x3f);
            resArr[(resPos += 1)] = (0x2 << 6) | ((point >>> 6) & 0x3f);
            resArr[(resPos += 1)] = (0x2 << 6) | (point & 0x3f);
            continue;
          }
        } else {
          resArr[(resPos += 1)] = 0xef;
          resArr[(resPos += 1)] = 0xbf;
          resArr[(resPos += 1)] = 0xbd;
          continue;
        }
      }
      if (point <= 0x007f) {
        resArr[(resPos += 1)] = (0x0 << 7) | point;
      } else if (point <= 0x07ff) {
        resArr[(resPos += 1)] = (0x6 << 5) | (point >>> 6);
        resArr[(resPos += 1)] = (0x2 << 6) | (point & 0x3f);
      } else {
        resArr[(resPos += 1)] = (0xe << 4) | (point >>> 12);
        resArr[(resPos += 1)] = (0x2 << 6) | ((point >>> 6) & 0x3f);
        resArr[(resPos += 1)] = (0x2 << 6) | (point & 0x3f);
      }
    }
    if (typeof Uint8Array !== 'undefined') {
      return new Uint8Array(resArr.buffer.slice(0, resPos + 1));
    } else {
      return resArr.length === resPos + 1 ? resArr : resArr.slice(0, resPos + 1);
    }
  };
  TextEncoder.prototype.toString = function () {
    return '[object TextEncoder]';
  };
  try {
    Object.defineProperty(TextEncoder.prototype, 'encoding', {
      get: function () {
        if (Object.prototype.isPrototypeOf.call(TextEncoder.prototype, this)) {
          return 'utf-8';
        } else {
          throw TypeError('Illegal invocation');
        }
      },
    });
  } catch (e) {
    TextEncoder.prototype.encoding = 'utf-8';
  }
  if (typeof Symbol !== 'undefined') {
    TextEncoder.prototype[Symbol.toStringTag] = 'TextEncoder';
  }
}

if (typeof URLSearchParams === 'undefined') {
  URLSearchParams = function URLSearchParams(search) {
    const decode = (str) => {
      return str.replace(/[ +]/g, '%20').replace(/(%[a-f0-9]{2})+/gi, (match) => {
        return decodeURIComponent(match);
      });
    };
    this._dict = {};
    if (typeof search === 'string') {
      search = search.indexOf('?') === 0 ? search.substring(1) : search;
      const properties = search.split('&');
      for (const property of properties) {
        const index = property.indexOf('=');
        const name = index > -1 ? decode(property.substring(0, index)) : decode(property);
        const value = index > -1 ? decode(property.substring(index + 1)) : '';
        if (!Object.prototype.hasOwnProperty.call(this._dict, name)) {
          this._dict[name] = [];
        }
        this._dict[name].push(value);
      }
    }
  };
  URLSearchParams.prototype.get = function (name) {
    return Object.prototype.hasOwnProperty.call(this._dict, name) ? this._dict[name][0] : null;
  };
}

if (!HTMLCanvasElement.prototype.toBlob) {
  HTMLCanvasElement.prototype.toBlob = function (callback, type, quality) {
    const canvas = this;
    setTimeout(function () {
      const data = atob(canvas.toDataURL(type, quality).split(',')[1]);
      const length = data.length;
      const buffer = new Uint8Array(length);
      for (let i = 0; i < length; i++) {
        buffer[i] = data.charCodeAt(i);
      }
      callback(new Blob([buffer], { type: type || 'image/png' }));
    });
  };
}

if (!('scrollBehavior' in window.document.documentElement.style)) {
  const __scrollTo__ = Element.prototype.scrollTo;
  Element.prototype.scrollTo = function (options) {
    if (options === undefined) {
      return;
    }
    if (
      options === null ||
      typeof options !== 'object' ||
      options.behavior === undefined ||
      arguments[0].behavior === 'auto' ||
      options.behavior === 'instant'
    ) {
      if (__scrollTo__) {
        __scrollTo__.apply(this, arguments);
      }
      return;
    }
    const now = () => {
      return window.performance && window.performance.now ? window.performance.now() : Date.now();
    };
    const ease = (k) => {
      return 0.5 * (1 - Math.cos(Math.PI * k));
    };
    const step = (context) => {
      const value = ease(Math.min((now() - context.startTime) / 468, 1));
      const x = context.startX + (context.x - context.startX) * value;
      const y = context.startY + (context.y - context.startY) * value;
      context.element.scrollLeft = x;
      context.element.scrollTop = y;
      if (x !== context.x || y !== context.y) {
        window.requestAnimationFrame(step.bind(window, context));
      }
    };
    const context = {
      element: this,
      x: typeof options.left === 'undefined' ? this.scrollLeft : ~~options.left,
      y: typeof options.top === 'undefined' ? this.scrollTop : ~~options.top,
      startX: this.scrollLeft,
      startY: this.scrollTop,
      startTime: now(),
    };
    step(context);
  };
}

window.addEventListener('load', () => {
  window.__view__ = new view.View(new host.BrowserHost());
});
