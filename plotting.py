import numpy as np
import matplotlib.pyplot as plt


def plot_result(returns, ymax_suggested=None, sample_rate=1, x_log_scale=False, cumulative=False):
    '''Exibe um gráfico "episódio/passo x retorno"
    
    Parâmetros:
    - returns: se return_type=='episode', este parâmetro é uma lista de retornos a cada episódio; se return_type=='step', é uma lista de pares (passo,retorno) 
    - ymax_suggested (opcional): valor máximo de retorno (eixo y), se tiver um valor máximo conhecido previamente
    - x_log_scale: se for True, mostra o eixo x na escala log (para detalhar mais os resultados iniciais)
    - return_type: use 'episode' ou 'step' para indicar o que representa o eixo x; também afeta como será lido o parâmetro 'returns'
    - filename: se for fornecida uma string, salva um arquivo de imagem ao invés de exibir.
    '''
    plt.figure(figsize=(12,7))

    if cumulative:
        returns = np.array(returns)
        returns = np.cumsum(returns)
    yvalues = returns
    xvalues = np.arange(1, len(returns)+1)
    plt.plot(xvalues, yvalues)

    if sample_rate == 1:
        plt.xlabel('Episódios')
        plt.title(f"Retorno médio a cada episódio")
    else:
        plt.xlabel(f'Episódios (x = {sample_rate} episódios)')
        plt.title(f"Retorno médio a cada {sample_rate} episódios")

    if x_log_scale:
        plt.xscale('log')

    plt.ylabel('Retorno')
    if ymax_suggested is not None:
        ymax = np.max([ymax_suggested, np.max(yvalues)])
        plt.ylim(top=ymax)

    plt.show()
    
    plt.close()