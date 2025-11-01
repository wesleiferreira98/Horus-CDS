#!/usr/bin/env python3
"""
Script de Teste de Configuração GPU/CUDA para Horus-CDS

Este script verifica se a instalação do CUDA, cuDNN e TensorFlow está
configurada corretamente para utilização de GPU no Horus-CDS.

Uso:
    python test_gpu_setup.py
"""

import sys
import platform

def print_section(title):
    """Imprime título de seção formatado"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_python_version():
    """Verifica versão do Python"""
    print_section("1. Verificação da Versão do Python")
    version = sys.version_info
    print(f"Versão do Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print("✓ Versão do Python compatível")
        return True
    else:
        print("✗ Versão do Python incompatível. Requer Python 3.9 ou superior")
        return False

def check_tensorflow():
    """Verifica instalação do TensorFlow"""
    print_section("2. Verificação do TensorFlow")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow versão: {tf.__version__}")
        print("✓ TensorFlow instalado corretamente")
        return True, tf
    except ImportError as e:
        print(f"✗ TensorFlow não encontrado: {e}")
        print("Execute: pip install tensorflow")
        return False, None

def check_cuda_support(tf):
    """Verifica suporte CUDA no TensorFlow"""
    print_section("3. Verificação do Suporte CUDA")
    
    try:
        cuda_built = tf.test.is_built_with_cuda()
        print(f"TensorFlow compilado com CUDA: {cuda_built}")
        
        if cuda_built:
            print("✓ TensorFlow possui suporte CUDA")
            return True
        else:
            print("✗ TensorFlow não possui suporte CUDA")
            print("Instalação CPU detectada. Para usar GPU, instale tensorflow-gpu")
            return False
    except Exception as e:
        print(f"✗ Erro ao verificar suporte CUDA: {e}")
        return False

def check_gpu_devices(tf):
    """Verifica dispositivos GPU disponíveis"""
    print_section("4. Detecção de Dispositivos GPU")
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("✗ Nenhuma GPU detectada")
            print("\nPossíveis causas:")
            print("  - Drivers NVIDIA não instalados")
            print("  - CUDA Toolkit não instalado")
            print("  - cuDNN não instalado")
            print("  - GPU não compatível")
            print(f"\nConsulte: CUDA_INSTALLATION.md")
            return False
        
        print(f"✓ {len(gpus)} GPU(s) detectada(s):\n")
        
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                if details:
                    compute_cap = details.get('compute_capability', 'N/A')
                    print(f"  Compute Capability: {compute_cap}")
                    
                    # Verificar se compute capability é suficiente
                    if compute_cap != 'N/A':
                        major, minor = compute_cap
                        if major >= 3 and minor >= 5:
                            print(f"  ✓ Compute Capability adequada para TensorFlow")
                        else:
                            print(f"  ✗ Compute Capability muito baixa (requer 3.5+)")
            except Exception as e:
                print(f"  Detalhes não disponíveis: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Erro ao listar dispositivos GPU: {e}")
        return False

def check_gpu_memory(tf):
    """Verifica memória GPU disponível"""
    print_section("5. Verificação de Memória GPU")
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            return False
        
        # Configurar crescimento de memória
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Crescimento de memória habilitado para {gpu.name}")
            except RuntimeError as e:
                print(f"Aviso: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Erro ao configurar memória GPU: {e}")
        return False

def benchmark_performance(tf):
    """Executa benchmark simples de performance"""
    print_section("6. Benchmark de Performance (CPU vs GPU)")
    
    try:
        import time
        import numpy as np
        
        print("Executando multiplicação de matrizes 5000x5000...")
        
        # Teste CPU
        with tf.device('/CPU:0'):
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            
            start = time.time()
            c = tf.matmul(a, b)
            c.numpy()  # Força execução
            cpu_time = time.time() - start
            
            print(f"Tempo CPU: {cpu_time:.4f}s")
        
        # Teste GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            with tf.device('/GPU:0'):
                a = tf.random.normal([5000, 5000])
                b = tf.random.normal([5000, 5000])
                
                start = time.time()
                c = tf.matmul(a, b)
                c.numpy()  # Força execução
                gpu_time = time.time() - start
                
                print(f"Tempo GPU: {gpu_time:.4f}s")
                
                speedup = cpu_time / gpu_time
                print(f"\n✓ Aceleração GPU: {speedup:.2f}x")
                
                if speedup > 1.5:
                    print("GPU está funcionando corretamente!")
                elif speedup > 1.0:
                    print("Aviso: Aceleração baixa. Verifique drivers e CUDA.")
                else:
                    print("Aviso: GPU mais lenta que CPU. Problema de configuração.")
        else:
            print("GPU não disponível para benchmark")
        
        return True
    except Exception as e:
        print(f"✗ Erro durante benchmark: {e}")
        return False

def check_additional_libraries():
    """Verifica bibliotecas adicionais necessárias"""
    print_section("7. Verificação de Bibliotecas Adicionais")
    
    libraries = {
        'keras-tcn': 'keras_tcn',
        'PyQt5': 'PyQt5',
        'Flask': 'flask',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'scapy': 'scapy'
    }
    
    all_ok = True
    for name, import_name in libraries.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'N/A')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: Não instalado")
            all_ok = False
    
    return all_ok

def print_system_info():
    """Imprime informações do sistema"""
    print_section("Informações do Sistema")
    
    print(f"Sistema Operacional: {platform.system()} {platform.release()}")
    print(f"Plataforma: {platform.platform()}")
    print(f"Arquitetura: {platform.machine()}")
    print(f"Processador: {platform.processor()}")

def print_recommendations(gpu_available):
    """Imprime recomendações baseadas nos resultados"""
    print_section("Recomendações")
    
    if gpu_available:
        print("✓ Sistema configurado corretamente para uso de GPU!")
        print("\nVocê pode:")
        print("  - Executar treinamentos com aceleração GPU")
        print("  - Usar o Horus-CDS com performance máxima")
        print("  - Experimentar com modelos mais complexos")
    else:
        print("Sistema configurado para uso apenas de CPU")
        print("\nPara habilitar GPU:")
        print("  1. Verifique se possui GPU NVIDIA compatível")
        print("  2. Instale drivers NVIDIA")
        print("  3. Instale CUDA Toolkit 12.3+")
        print("  4. Instale cuDNN 9.0+")
        print("  5. Reinstale TensorFlow com suporte GPU")
        print(f"\nConsulte o guia completo: CUDA_INSTALLATION.md")

def main():
    """Função principal"""
    print("=" * 70)
    print(" TESTE DE CONFIGURAÇÃO GPU/CUDA - HORUS-CDS")
    print("=" * 70)
    
    print_system_info()
    
    # Verificações
    python_ok = check_python_version()
    if not python_ok:
        print("\n✗ Teste abortado: Python incompatível")
        sys.exit(1)
    
    tf_ok, tf = check_tensorflow()
    if not tf_ok:
        print("\n✗ Teste abortado: TensorFlow não instalado")
        sys.exit(1)
    
    cuda_ok = check_cuda_support(tf)
    gpu_ok = check_gpu_devices(tf)
    
    gpu_available = cuda_ok and gpu_ok
    
    if gpu_available:
        check_gpu_memory(tf)
        benchmark_performance(tf)
    
    libs_ok = check_additional_libraries()
    
    print_recommendations(gpu_available)
    
    print_section("Resumo Final")
    
    print(f"Python: {'✓' if python_ok else '✗'}")
    print(f"TensorFlow: {'✓' if tf_ok else '✗'}")
    print(f"Suporte CUDA: {'✓' if cuda_ok else '✗'}")
    print(f"GPU Detectada: {'✓' if gpu_ok else '✗'}")
    print(f"Bibliotecas: {'✓' if libs_ok else '✗'}")
    
    if python_ok and tf_ok and libs_ok:
        if gpu_available:
            print("\n✓ SISTEMA PRONTO PARA USO COM GPU")
            sys.exit(0)
        else:
            print("\n⚠ SISTEMA PRONTO PARA USO COM CPU APENAS")
            sys.exit(0)
    else:
        print("\n✗ SISTEMA COM PROBLEMAS DE CONFIGURAÇÃO")
        sys.exit(1)

if __name__ == "__main__":
    main()
