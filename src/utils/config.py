from simple_term_menu import TerminalMenu
import json
import os
from pygments import formatters, highlight, lexers
from pygments.util import ClassNotFound
from simple_term_menu import TerminalMenu
from . import paths
#mport paths

configName = "default_config.json"
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), configName)

## read_config
def readConfig():
    with open(config_path, 'r') as configfile:
        config = json.load(configfile)
    return(config)

## write_config
def writeConfig(dataset,clusterset,model):
    config = {
    'dataset': dataset,
    'clusterset': clusterset,
    'model':model,
    }
    with open(config_path, 'w') as configfile:
        json.dump(config, configfile)

def highlight_file(filepath):
    with open(filepath, "r", errors='ignore') as f:
        file_content = f.read()
    try:
        lexer = lexers.get_lexer_for_filename(filepath, stripnl=False, stripall=False)
    except ClassNotFound:
        lexer = lexers.get_lexer_by_name("text", stripnl=False, stripall=False)
    formatter = formatters.TerminalFormatter(bg="dark")  # dark or light
    highlighted_file_content = highlight(file_content, lexer, formatter)
    return highlighted_file_content

def list_files(directory):
    return (os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)))

def setConfig():
    datasetPaths = paths.list_directories(paths.data)
    clustersetPaths = paths.list_directories(paths.clustering_logs)
    modelPaths = paths.list_files(paths.models)
    
    terminal_menu = TerminalMenu(list_files(paths.models), preview_command=highlight_file, preview_size=0.75)
    menu_entry_index = terminal_menu.show()
    model = modelPaths[terminal_menu.show()+1]
    print(f"model: {model}")

    terminal_menu = TerminalMenu(datasetPaths, title="available datasets")
    dataset = datasetPaths[terminal_menu.show()]
    print(f"dataset: {dataset}")

    terminal_menu = TerminalMenu(clustersetPaths, title="available clustersets")
    clusterset = clustersetPaths[terminal_menu.show()]
    print(f"clusterset: {clusterset}")
   
    writeConfig(dataset,clusterset,model)

#setConfig()
#readConfig()
