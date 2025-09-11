# -*- coding: utf-8 -*-

def cortar_arquivo_txt(arquivo_entrada, arquivo_saida, porcentagem_a_manter):
    """
    Lê um arquivo de texto e salva um novo arquivo contendo apenas uma 
    porcentagem inicial das linhas do original.

    Args:
        arquivo_entrada (str): O caminho para o arquivo de texto original.
        arquivo_saida (str): O caminho para o novo arquivo que será criado.
        porcentagem_a_manter (float): A porcentagem de linhas do início 
                                      do arquivo que devem ser mantidas (0 a 100).
    """
    if not 0 <= porcentagem_a_manter <= 100:
        print("Erro: A porcentagem deve estar entre 0 e 100.")
        return

    try:
        # Abre o arquivo de entrada para leitura ('r') com codificação utf-8
        with open(arquivo_entrada, 'r', encoding='utf-8') as f_in:
            # Lê todas as linhas do arquivo e as armazena em uma lista
            linhas = f_in.readlines()

        # Calcula o número total de linhas
        total_de_linhas = len(linhas)
        
        # Calcula o número de linhas que devem ser mantidas
        # Usa int() para garantir que o resultado seja um número inteiro
        linhas_a_manter = int(total_de_linhas * (porcentagem_a_manter / 100))

        # Seleciona apenas as linhas calculadas, do início até o limite
        linhas_cortadas = linhas[:linhas_a_manter]

        # Abre o arquivo de saída para escrita ('w')
        with open(arquivo_saida, 'w', encoding='utf-8') as f_out:
            # Escreve as linhas selecionadas no novo arquivo
            f_out.writelines(linhas_cortadas)

        print("\nProcesso concluído com sucesso!")
        print(f"  - Arquivo de entrada: '{arquivo_entrada}' ({total_de_linhas} linhas)")
        print(f"  - Arquivo de saída: '{arquivo_saida}' ({len(linhas_cortadas)} linhas)")
        print(f"  - Foram mantidos os primeiros {porcentagem_a_manter}% do arquivo.")

    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada '{arquivo_entrada}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# --- Como Usar ---
if __name__ == "__main__":
    # 1. Defina o nome do arquivo original
    arquivo_original = "weatherAUS_rainfall_prediction.csv"
    
    # 2. Defina o nome do novo arquivo que será gerado
    arquivo_cortado = "weatherAUS_reduced_05.csv"
    
    # 3. Defina a porcentagem de linhas que você quer MANTER (ex: 25.5 para 25.5%)
    porcentagem = 5
            
    # Chama a função para executar a tarefa
    cortar_arquivo_txt(arquivo_original, arquivo_cortado, porcentagem)