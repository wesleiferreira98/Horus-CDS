import os

# Variável global para controlar uso de GPU
USE_GPU = True

def set_tf_environment():
    """Configura variaveis de ambiente para TensorFlow (DEVE SER CHAMADO ANTES DE IMPORTAR TF)"""
    
    # Verificar se usuário quer forçar CPU
    force_cpu = os.environ.get('HORUS_FORCE_CPU', '0') == '1'
    
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("=" * 60)
        print("MODO CPU FORÇADO (via HORUS_FORCE_CPU=1)")
        print("=" * 60)
        return
    
    # Reduzir verbosidade do TensorFlow (0=todos, 1=info, 2=warning, 3=error)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Habilitar otimizacoes oneDNN se disponivel
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    
    # Garantir que o TensorFlow possa encontrar CUDA
    cuda_path = '/usr/local/cuda-13.0'
    if os.path.exists(cuda_path):
        os.environ['CUDA_HOME'] = cuda_path
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

# CONFIGURAR AMBIENTE ANTES DE IMPORTAR TENSORFLOW
set_tf_environment()

# Agora sim importar TensorFlow
import tensorflow as tf

def configure_gpu(force_cpu=False):
    """Configura GPU para uso com TensorFlow"""
    
    global USE_GPU
    
    if force_cpu or os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
        print("=" * 60)
        print("MODO CPU")
        print("GPU desabilitada")
        print("=" * 60)
        USE_GPU = False
        return False
    
    # Verificar GPUs disponiveis
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Permitir crescimento dinamico de memoria (evita alocar toda memoria de uma vez)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print("=" * 60)
            print("CONFIGURACAO DE GPU")
            print("=" * 60)
            print(f"GPUs físicas detectadas: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print(f"GPUs lógicas configuradas: {len(logical_gpus)}")
            print("GPU será utilizada para treinamento!")
            print("=" * 60)
            USE_GPU = True
            return True
        except RuntimeError as e:
            print(f"Erro ao configurar GPU: {e}")
            print("Usando CPU como fallback.")
            USE_GPU = False
            return False
    else:
        print("=" * 60)
        print("Nenhuma GPU disponível detectada.")
        print("Usando CPU para processamento.")
        print("=" * 60)
        USE_GPU = False
        return False

def has_gpu():
    """Verifica se há GPU disponível no sistema"""
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
        return False
    gpus = tf.config.list_physical_devices('GPU')
    return len(gpus) > 0

def is_using_gpu():
    """Retorna se GPU está sendo usada atualmente"""
    return USE_GPU and has_gpu()

# Configurar ao importar
set_tf_environment()
gpu_available = configure_gpu()
