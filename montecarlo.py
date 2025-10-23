import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cholesky
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import pickle
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ConfiguracionMonteCarlo:
    activos: List[str]
    n_simulaciones: int = 1000
    n_pasos: int = 252  
    tiempo_anos: float = 1.0
    intensidad_contagio: float = 0.15
    umbral_shock: float = 2.0  
    semilla_aleatoria: Optional[int] = 42
    incluir_contagio: bool = True

@dataclass  
class ParametrosFinancieros:
    precios_iniciales: np.ndarray
    rendimientos_esperados: np.ndarray
    volatilidades: np.ndarray
    matriz_correlacion: np.ndarray
    
    def validar(self):
        n = len(self.precios_iniciales)
        assert len(self.rendimientos_esperados) == n
        assert len(self.volatilidades) == n
        assert self.matriz_correlacion.shape == (n, n)
        
        
        eigenvals = np.linalg.eigvals(self.matriz_correlacion)
        if not np.all(eigenvals >= -1e-8):
            eigenvals = np.maximum(eigenvals, 0.01)
            eigenvecs = np.linalg.eigh(self.matriz_correlacion)[1]
            self.matriz_correlacion = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            D = np.sqrt(np.diag(self.matriz_correlacion))
            self.matriz_correlacion = self.matriz_correlacion / np.outer(D, D)

class ResultadosMonteCarlo:
    def __init__(self, config: ConfiguracionMonteCarlo, parametros: ParametrosFinancieros):
        self.config = config
        self.parametros = parametros
        
        
        self.precios_simulados = None  
        self.shocks_temporales = None  
        self.eventos_contagio = None   
        self.metricas_riesgo = None    
        self.matriz_contagio = None    
        
        
        self.timestamp = datetime.now().isoformat()
    
    def get_datos_componente_2(self) -> Dict:
        if self.precios_simulados is None:
            raise ValueError("Debe ejecutar simulación primero")
        
        umbral_correlacion = 0.3
        matriz_adyacencia = (np.abs(self.parametros.matriz_correlacion) > umbral_correlacion).astype(int)
        np.fill_diagonal(matriz_adyacencia, 0)
        
        
        rendimientos = np.diff(np.log(self.precios_simulados), axis=1)
        vol_realizada = np.std(rendimientos, axis=(0,1))
        centralidad_volatilidad = vol_realizada / np.max(vol_realizada)
        
        
        centralidad_contagio = np.sum(self.matriz_contagio, axis=1)
        
        return {
            'activos': self.config.activos,
            'matriz_correlacion': self.parametros.matriz_correlacion,
            'matriz_adyacencia': matriz_adyacencia,
            'pesos_aristas': np.abs(self.parametros.matriz_correlacion) * matriz_adyacencia,
            'centralidad_volatilidad': centralidad_volatilidad,
            'centralidad_contagio': centralidad_contagio,
            'parametros_barabasi_albert': {
                'n_nodos': len(self.config.activos),
                'm_conexiones': max(1, int(np.sum(matriz_adyacencia) / len(self.config.activos) / 2)),
                'seed': self.config.semilla_aleatoria
            }
        }
    
    def get_datos_componente_3(self) -> Dict:
        if self.shocks_temporales is None:
            raise ValueError("Debe ejecutar simulación primero")
        
        
        centralidad_vol = self.get_datos_componente_2()['centralidad_volatilidad']
        umbral_inicial = np.percentile(centralidad_vol, 80)
        
        susceptibles = (centralidad_vol <= umbral_inicial).astype(float)
        infectados = (centralidad_vol > umbral_inicial).astype(float)
        recuperados = np.zeros(len(self.config.activos))
        
        
        contador_origen = {}
        for evento in self.eventos_contagio:
            for activo_idx in evento['activos_origen']:
                activo = self.config.activos[activo_idx]
                contador_origen[activo] = contador_origen.get(activo, 0) + 1
        
        activos_ordenados = sorted(contador_origen.items(), key=lambda x: x[1], reverse=True)
        n_super = max(1, len(activos_ordenados) // 5)
        superpropagadores = [activo for activo, _ in activos_ordenados[:n_super]]
        
        return {
            'shocks_temporales': self.shocks_temporales,
            'eventos_contagio': self.eventos_contagio,
            'matriz_transmision': self.matriz_contagio,
            'estados_sir_iniciales': {
                'S': susceptibles,
                'I': infectados,
                'R': recuperados
            },
            'superpropagadores': superpropagadores,
            'parametros_sir': {
                'beta': self.config.intensidad_contagio,
                'gamma': 0.1,  
                'umbral_infeccion': self.config.umbral_shock
            }
        }
    
    def get_datos_componente_4(self) -> Dict:
        if self.precios_simulados is None:
            raise ValueError("Debe ejecutar simulación primero")
        
        
        precios_promedio = np.mean(self.precios_simulados, axis=0)
        
        
        timestamps = pd.date_range(start='2024-01-01', periods=len(precios_promedio), freq='D')
        df_precios = pd.DataFrame(precios_promedio, index=timestamps, columns=self.config.activos)
        
        
        df_rendimientos = df_precios.pct_change().fillna(0)
        df_volatilidad = df_rendimientos.rolling(21).std().fillna(0)
        
        
        ventanas = [5, 10, 21, 63]
        vol_realizadas_dict = {}
        for ventana in ventanas:
            vol_ventana = df_rendimientos.rolling(ventana).std() * np.sqrt(252)
            vol_realizadas_dict[f'vol_{ventana}d'] = vol_ventana.mean(axis=1)  
        
        df_vol_realizadas = pd.DataFrame(vol_realizadas_dict, index=timestamps)
        
        
        vol_promedio = df_vol_realizadas.mean(axis=1)
        umbrales = [np.percentile(vol_promedio.dropna(), p) for p in [25, 75, 95]]
        
        regimenes = []
        for vol in vol_promedio:
            if pd.isna(vol):
                regimenes.append('Normal')  
            elif vol <= umbrales[0]:
                regimenes.append('Bajo_Vol')
            elif vol <= umbrales[1]:
                regimenes.append('Normal')
            elif vol <= umbrales[2]:
                regimenes.append('Alto_Vol')
            else:
                regimenes.append('Crisis')
        
        
        features_ml = self._calcular_features_ml(df_precios, df_rendimientos)
        
        return {
            'series_temporales': {
                'precios': df_precios,
                'rendimientos': df_rendimientos,
                'volatilidad': df_volatilidad,
                'timestamps': timestamps
            },
            'volatilidades_realizadas': df_vol_realizadas,
            'regimenes_mercado': pd.DataFrame({
                'regimen': regimenes,
                'volatilidad_promedio': vol_promedio
            }, index=timestamps),
            'features_ml': features_ml,
            'parametros_lstm': {
                'secuencia_lookback': 60,
                'horizonte_prediccion': 20,
                'features_input': ['precio', 'rendimiento', 'volatilidad']
            },
            'parametros_clasificador': {
                'clases_regimen': ['Bajo_Vol', 'Normal', 'Alto_Vol', 'Crisis'],
                'umbrales_volatilidad': umbrales,
                'ventana_clasificacion': 21
            }
        }
    
    def _calcular_features_ml(self, precios: pd.DataFrame, rendimientos: pd.DataFrame) -> pd.DataFrame:
        features = {}
        
        
        features['rsi_promedio'] = self._calcular_rsi(precios).mean(axis=1)
        
        
        ma_5 = precios.rolling(5).mean().mean(axis=1)
        ma_21 = precios.rolling(21).mean().mean(axis=1)
        features['ma_ratio'] = ma_5 / ma_21
        
        
        features['correlacion_promedio'] = self._correlacion_rolling(rendimientos)
        
        
        features['dispersion'] = rendimientos.std(axis=1)
        
        
        features['momentum_5d'] = (precios / precios.shift(5) - 1).mean(axis=1)
        features['momentum_21d'] = (precios / precios.shift(21) - 1).mean(axis=1)
        
        return pd.DataFrame(features).fillna(0)
    
    def _calcular_rsi(self, precios: pd.DataFrame, periodo: int = 14) -> pd.DataFrame:
        delta = precios.diff()
        ganancia = delta.where(delta > 0, 0)
        perdida = -delta.where(delta < 0, 0)
        
        avg_ganancia = ganancia.rolling(periodo).mean()
        avg_perdida = perdida.rolling(periodo).mean()
        
        rs = avg_ganancia / avg_perdida
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _correlacion_rolling(self, rendimientos: pd.DataFrame, ventana: int = 21) -> pd.Series:
        correlaciones = []
        
        for i in range(ventana, len(rendimientos)):
            ventana_datos = rendimientos.iloc[i-ventana:i]
            matriz_corr = ventana_datos.corr()
            triu_indices = np.triu_indices_from(matriz_corr, k=1)
            corr_promedio = matriz_corr.values[triu_indices].mean()
            correlaciones.append(corr_promedio)
        
        serie_completa = [0] * ventana + correlaciones
        return pd.Series(serie_completa, index=rendimientos.index)
    
    def guardar_resultados(self, directorio: str = "./") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_completa = os.path.join(directorio, f"componente_1_resultados_{timestamp}")
        os.makedirs(ruta_completa, exist_ok=True)
        
        
        config_data = {
            'configuracion': asdict(self.config),
            'parametros': {
                'precios_iniciales': self.parametros.precios_iniciales.tolist(),
                'rendimientos_esperados': self.parametros.rendimientos_esperados.tolist(),
                'volatilidades': self.parametros.volatilidades.tolist(),
                'matriz_correlacion': self.parametros.matriz_correlacion.tolist()
            },
            'metricas_riesgo': self.metricas_riesgo,
            'timestamp': self.timestamp
        }
        
        with open(os.path.join(ruta_completa, 'configuracion.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        
        
        datos_simulacion = {
            'precios_simulados': self.precios_simulados,
            'shocks_temporales': self.shocks_temporales,
            'eventos_contagio': self.eventos_contagio,
            'matriz_contagio': self.matriz_contagio
        }
        
        with open(os.path.join(ruta_completa, 'datos_simulacion.pkl'), 'wb') as f:
            pickle.dump(datos_simulacion, f)
        
        
        for i, metodo in enumerate([self.get_datos_componente_2, self.get_datos_componente_3, self.get_datos_componente_4], 2):
            try:
                datos = metodo()
                with open(os.path.join(ruta_completa, f'datos_componente_{i}.pkl'), 'wb') as f:
                    pickle.dump(datos, f)
            except Exception as e:
                print(f"Error guardando datos componente {i}: {e}")
        
        return ruta_completa





class SimuladorMonteCarlo:
    def __init__(self, config: Optional[ConfiguracionMonteCarlo] = None):

        self.config = config
        self.parametros = None
        self.resultados = None
        
        if config and config.semilla_aleatoria:
            np.random.seed(config.semilla_aleatoria)
    
    def crear_datos_ejemplo(self) -> Tuple[ConfiguracionMonteCarlo, ParametrosFinancieros]:        
        activos = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'GLD',      
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
            'JPM', 'BAC', 'WFC', 'GS', 'MS',         
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',     
            'XOM', 'CVX', 'COP', 'SLB', 'EOG'       
        ]
        
        n_activos = len(activos)
        np.random.seed(42)
        
        
        precios_iniciales = np.random.uniform(50, 500, n_activos)
        rendimientos_esperados = np.random.uniform(-0.05, 0.15, n_activos)
        volatilidades = np.random.uniform(0.15, 0.60, n_activos)
        
        
        correlacion = np.eye(n_activos)
        
        sectores = {
            'ETFs': range(0, 5),
            'Tech': range(5, 10),
            'Financieros': range(10, 15),
            'Salud': range(15, 20),
            'Energía': range(20, 25)
        }
        
        
        for indices in sectores.values():
            for i in indices:
                for j in indices:
                    if i != j:
                        correlacion[i, j] = np.random.uniform(0.4, 0.8)
        
        
        for i in range(n_activos):
            for j in range(n_activos):
                if correlacion[i, j] == 0:
                    correlacion[i, j] = np.random.uniform(-0.2, 0.3)
                    correlacion[j, i] = correlacion[i, j]
        
        
        eigenvals, eigenvecs = np.linalg.eigh(correlacion)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlacion = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        D = np.sqrt(np.diag(correlacion))
        correlacion = correlacion / np.outer(D, D)
        
        
        config = ConfiguracionMonteCarlo(activos=activos) if self.config is None else self.config
        
        
        parametros = ParametrosFinancieros(
            precios_iniciales=precios_iniciales,
            rendimientos_esperados=rendimientos_esperados,
            volatilidades=volatilidades,
            matriz_correlacion=correlacion
        )
        parametros.validar()
        
        return config, parametros
    
    def _crear_matriz_contagio(self, correlacion: np.ndarray) -> np.ndarray:
        matriz = np.abs(correlacion)
        np.fill_diagonal(matriz, 0)
        
        
        row_sums = matriz.sum(axis=1)
        row_sums[row_sums == 0] = 1
        matriz = matriz / row_sums[:, np.newaxis]
        
        return matriz
    
    def _aplicar_contagio(self, shocks: np.ndarray, matriz_contagio: np.ndarray) -> np.ndarray:
        shocks_modificados = shocks.copy()
        n_activos = len(shocks)
        
        
        activos_shock = np.abs(shocks) > self.config.umbral_shock
        
        for i in range(n_activos):
            if activos_shock[i]:
                for j in range(n_activos):
                    if i != j and matriz_contagio[i, j] > 0:
                        contagio = (matriz_contagio[i, j] * 
                                  self.config.intensidad_contagio * 
                                  shocks[i])
                        shocks_modificados[j] += contagio
        
        return shocks_modificados
    
    def _detectar_eventos_contagio(self, shocks_temporales: np.ndarray) -> List[Dict]:
        eventos = []
        
        for sim in range(shocks_temporales.shape[0]):
            for t in range(shocks_temporales.shape[1]):
                shocks_extremos = np.abs(shocks_temporales[sim, t, :]) > self.config.umbral_shock
                
                if np.any(shocks_extremos):
                    activos_afectados = np.where(shocks_extremos)[0]
                    
                    evento = {
                        'simulacion': sim,
                        'tiempo': t,
                        'activos_origen': activos_afectados.tolist(),
                        'intensidad_maxima': np.max(np.abs(shocks_temporales[sim, t, :])),
                        'num_activos_afectados': len(activos_afectados)
                    }
                    eventos.append(evento)
        
        return eventos
    
    def _calcular_metricas_riesgo(self, precios: np.ndarray, pesos: np.ndarray = None) -> Dict:
        if pesos is None:
            pesos = np.ones(precios.shape[2]) / precios.shape[2]
        
        
        valor_portafolio = np.sum(precios * pesos, axis=2)
        valor_final = valor_portafolio[:, -1]
        valor_inicial = valor_portafolio[:, 0]
        
        
        rendimientos = (valor_final - valor_inicial) / valor_inicial
        
        
        return {
            'rendimiento_promedio': np.mean(rendimientos),
            'volatilidad': np.std(rendimientos),
            'var_95': np.percentile(rendimientos, 5),
            'var_99': np.percentile(rendimientos, 1),
            'cvar_95': np.mean(rendimientos[rendimientos <= np.percentile(rendimientos, 5)]),
            'prob_perdida': np.mean(rendimientos < 0),
            'sharpe_ratio': np.mean(rendimientos) / np.std(rendimientos) if np.std(rendimientos) > 0 else 0,
            'max_drawdown': self._calcular_max_drawdown(valor_portafolio)
        }
    
    def _calcular_max_drawdown(self, valor_portafolio: np.ndarray) -> float:
        drawdowns = []
        
        for sim in range(valor_portafolio.shape[0]):
            valores = valor_portafolio[sim, :]
            peak = np.maximum.accumulate(valores)
            drawdown = (valores - peak) / peak
            drawdowns.append(np.min(drawdown))
        
        return np.mean(drawdowns)
    
    def ejecutar_simulacion(self, 
                          config: Optional[ConfiguracionMonteCarlo] = None,
                          parametros: Optional[ParametrosFinancieros] = None,
                          usar_datos_ejemplo: bool = True) -> ResultadosMonteCarlo:
      
        
        if config is None and parametros is None and usar_datos_ejemplo:
            config, parametros = self.crear_datos_ejemplo()
        elif config is not None and parametros is None:
            raise ValueError("Se necesitan parametros para la simulacion")
        
        self.config = config or self.config
        self.parametros = parametros
        
        if self.config is None:
            raise ValueError("Debe proporcionar configuración")
        
        print(f"Simulando {len(self.config.activos)} activos")
        print(f"{self.config.n_simulaciones:,} simulaciones x {self.config.n_pasos:,} pasos")
        
        
        self.resultados = ResultadosMonteCarlo(self.config, self.parametros)
        
        
        matriz_contagio = self._crear_matriz_contagio(self.parametros.matriz_correlacion)
        self.resultados.matriz_contagio = matriz_contagio
        
        
        dt = self.config.tiempo_anos / self.config.n_pasos
        n_activos = len(self.config.activos)
        
        
        L = cholesky(self.parametros.matriz_correlacion, lower=True)
        
        
        precios = np.zeros((self.config.n_simulaciones, self.config.n_pasos + 1, n_activos))
        shocks_temporales = np.zeros((self.config.n_simulaciones, self.config.n_pasos, n_activos))
        
        
        precios[:, 0, :] = self.parametros.precios_iniciales
        
                
        for sim in range(self.config.n_simulaciones):
            for t in range(1, self.config.n_pasos + 1):
                
                shocks_independientes = np.random.standard_normal(n_activos)
                
                
                shocks_correlacionados = L @ shocks_independientes
                
                
                if self.config.incluir_contagio:
                    shocks_finales = self._aplicar_contagio(shocks_correlacionados, matriz_contagio)
                else:
                    shocks_finales = shocks_correlacionados
                
                
                shocks_temporales[sim, t-1, :] = shocks_finales
                
                
                drift = (self.parametros.rendimientos_esperados - 0.5 * self.parametros.volatilidades**2) * dt
                difusion = self.parametros.volatilidades * np.sqrt(dt) * shocks_finales
                
                precios[sim, t, :] = precios[sim, t-1, :] * np.exp(drift + difusion)
        
        
        self.resultados.precios_simulados = precios
        self.resultados.shocks_temporales = shocks_temporales
        
        
        self.resultados.eventos_contagio = self._detectar_eventos_contagio(shocks_temporales)
        
        
        self.resultados.metricas_riesgo = self._calcular_metricas_riesgo(precios)
        
        print(f"Eventos de contagio detectados: {len(self.resultados.eventos_contagio)}")
        print(f"VaR 95%: {self.resultados.metricas_riesgo['var_95']*100:.2f}%")
        
        return self.resultados
    
    def mostrar_resumen(self):
        if self.resultados is None:
            print("No hay resultados disponibles.")
            return
        
        print("Resultados obtenidos de Monte Carlo")
        
        print(f"\nCONFIGURACIÓN:")
        print(f"  Activos: {len(self.config.activos)}")
        print(f"  Simulaciones: {self.config.n_simulaciones:,}")
        print(f"  Horizonte: {self.config.tiempo_anos} años")
        print(f"  Contagio: {'Activado' if self.config.incluir_contagio else 'Desactivado'}")
        
        print(f"\nMétricas de reisgo:")
        metricas = self.resultados.metricas_riesgo
        for nombre, valor in metricas.items():
            if isinstance(valor, float):
                if 'ratio' in nombre.lower():
                    print(f"  {nombre.replace('_', ' ').title()}: {valor:.3f}")
                else:
                    print(f"  {nombre.replace('_', ' ').title()}: {valor*100:.2f}%")
        
        print(f"\nAnálisis de contagio:")
        print(f"  Eventos detectados: {len(self.resultados.eventos_contagio)}")
        if self.resultados.eventos_contagio:
            intensidades = [e['intensidad_maxima'] for e in self.resultados.eventos_contagio]
            print(f"  Intensidad promedio: {np.mean(intensidades):.2f}")
            print(f"  Intensidad máxima: {np.max(intensidades):.2f}")
        
    
    def visualizar_resultados(self):
        if self.resultados is None:
            print("Hubo un problema con la visualización de los resultados")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        
        precios = self.resultados.precios_simulados
        pesos = np.ones(precios.shape[2]) / precios.shape[2]
        valor_portafolio = np.sum(precios * pesos, axis=2)
        
        for i in range(min(100, precios.shape[0])):
            axes[0,0].plot(valor_portafolio[i], alpha=0.1, color='blue')
        
        valor_promedio = np.mean(valor_portafolio, axis=0)
        axes[0,0].plot(valor_promedio, 'red', linewidth=2, label='Promedio')
        axes[0,0].set_title('Evolución del Portafolio')
        axes[0,0].set_xlabel('Días')
        axes[0,0].set_ylabel('Valor')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        
        valor_final = valor_portafolio[:, -1]
        valor_inicial = valor_portafolio[:, 0]
        rendimientos = (valor_final - valor_inicial) / valor_inicial
        
        axes[0,1].hist(rendimientos * 100, bins=50, alpha=0.7, color='skyblue')
        axes[0,1].axvline(self.resultados.metricas_riesgo['var_95'] * 100, 
                         color='red', linestyle='--', label='VaR 95%')
        axes[0,1].set_title('Distribución de Rendimientos')
        axes[0,1].set_xlabel('Rendimiento (%)')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        
        corr = self.parametros.matriz_correlacion
        im = axes[1,0].imshow(corr, cmap='RdBu', vmin=-1, vmax=1)
        axes[1,0].set_title('Matriz de Correlación')
        plt.colorbar(im, ax=axes[1,0])
        
        
        if self.resultados.eventos_contagio:
            tiempos = [e['tiempo'] for e in self.resultados.eventos_contagio]
            intensidades = [e['intensidad_maxima'] for e in self.resultados.eventos_contagio]
            
            axes[1,1].scatter(tiempos, intensidades, alpha=0.6, color='orange')
            axes[1,1].axhline(y=self.config.umbral_shock, color='red', 
                             linestyle='--', label=f'Umbral ({self.config.umbral_shock})')
            axes[1,1].set_title('Eventos de Contagio')
            axes[1,1].set_xlabel('Tiempo (días)')
            axes[1,1].set_ylabel('Intensidad')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def ejecutar_simulacion_rapida(n_simulaciones: int = 1000, 
                              n_activos: int = 25,
                              incluir_contagio: bool = True) -> ResultadosMonteCarlo:
    simulador = SimuladorMonteCarlo()
    
    
    config, parametros = simulador.crear_datos_ejemplo()
    config.n_simulaciones = n_simulaciones
    config.incluir_contagio = incluir_contagio
    
    
    if n_activos < 25:
        config.activos = config.activos[:n_activos]
        parametros.precios_iniciales = parametros.precios_iniciales[:n_activos]
        parametros.rendimientos_esperados = parametros.rendimientos_esperados[:n_activos]
        parametros.volatilidades = parametros.volatilidades[:n_activos]
        parametros.matriz_correlacion = parametros.matriz_correlacion[:n_activos, :n_activos]
    
    
    resultados = simulador.ejecutar_simulacion(config=config, parametros=parametros, usar_datos_ejemplo=False)
    
    return resultados

def cargar_resultados(directorio: str) -> ResultadosMonteCarlo:
    
    with open(os.path.join(directorio, 'configuracion.json'), 'r') as f:
        config_data = json.load(f)
    
    config = ConfiguracionMonteCarlo(**config_data['configuracion'])
    
    parametros = ParametrosFinancieros(
        precios_iniciales=np.array(config_data['parametros']['precios_iniciales']),
        rendimientos_esperados=np.array(config_data['parametros']['rendimientos_esperados']),
        volatilidades=np.array(config_data['parametros']['volatilidades']),
        matriz_correlacion=np.array(config_data['parametros']['matriz_correlacion'])
    )
    
    
    resultados = ResultadosMonteCarlo(config, parametros)
    resultados.metricas_riesgo = config_data['metricas_riesgo']
    
    
    with open(os.path.join(directorio, 'datos_simulacion.pkl'), 'rb') as f:
        datos = pickle.load(f)
        resultados.precios_simulados = datos['precios_simulados']
        resultados.shocks_temporales = datos['shocks_temporales']
        resultados.eventos_contagio = datos['eventos_contagio']
        resultados.matriz_contagio = datos['matriz_contagio']
    
    return resultados


if __name__ == "__main__":
      
    simulador = SimuladorMonteCarlo()
    
    resultados = simulador.ejecutar_simulacion()
    
    simulador.mostrar_resumen()
    
    datos_redes = resultados.get_datos_componente_2()
    
    datos_contagio = resultados.get_datos_componente_3()
    
    datos_ia = resultados.get_datos_componente_4()
    
    directorio = resultados.guardar_resultados()
    
    simulador.visualizar_resultados()
