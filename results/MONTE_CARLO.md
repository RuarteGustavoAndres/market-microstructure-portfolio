# Monte Carlo del audit demo — tasas de FP/FN sobre semillas

Source: [`demo/monte_carlo_audit.py`](../demo/monte_carlo_audit.py) — corre el pipeline completo de [`demo/audit_demo.py`](../demo/audit_demo.py) (mismo market aleatorio sin edge, mismo rng compartido entre los dos clientes) variando la semilla del generador, y mide los errores del audit.

**Definiciones.** *FP*: el audit APRUEBA una estrategia rota — un check de la TRAMPA que debería disparar (FAIL/WARN) vuelve PASS. *FN*: el audit FLAGEA una estrategia honesta perdedora — un check del control limpio que debería pasar vuelve FAIL/WARN.

## Resumen

```
N = 30 runs (seeds 0..29), alpha=0.01, 999 permutaciones/check

RUN-LEVEL:  FP (trampa aprobada de punta a punta, 5/5 PASS) = 0/30
            FN (control honesto flaggeado por >=1 check)    = 1/30

DETECTOR-LEVEL (checks que FALLARON en su objetivo):
   check                  FP: trampa no cazada   FN: limpio flaggeado

   1 capturabilidad            0/30               0/30   
   2 fills al book             1/30               0/30   
   3 tuning in-sample          0/30               0/30   
   4 K-fold CV                 0/30               0/30   
   5 supervivencia             0/30               1/30   
```

## Runs individuales

Veredictos por check (F=FAIL, W=WARN, P=PASS, orden: 1 capturabilidad, 2 fills al book, 3 tuning in-sample, 4 K-fold CV, 5 supervivencia). Marco `*` = FP (la trampa pasó un check).

```
 seed  TRAP      CLEAN        Sh_trap  mean_clean
    0  WFFFF     PPPPP          98.02     -0.0081
    1  WFFFF     PPPPP          99.28     -0.0078
    2  WFFFF     PPPPP         102.58     -0.0080
    3  WFFFF     PPPPP         104.85     -0.0080
    4  WPFFF     PPPPP          98.02     -0.0085  *
    5  WFFFF     PPPPP         104.18     -0.0080
    6  WFFFF     PPPPP         108.43     -0.0080
    7  WFFFF     PPPPP          98.33     -0.0078
    8  WFFFF     PPPPP         100.47     -0.0079
    9  WFFFF     PPPPP          96.61     -0.0076
   10  WFFFF     PPPPP         101.97     -0.0079
   11  WFFFF     PPPPP         104.39     -0.0078
   12  WFFFF     PPPPP          94.37     -0.0077
   13  WFFFF     PPPPP         110.68     -0.0079
   14  WFFFF     PPPPP         105.94     -0.0086
   15  WFFFF     PPPPP         104.02     -0.0082
   16  WFFFF     PPPPP          96.34     -0.0080
   17  WFFFF     PPPPP         108.34     -0.0081
   18  WFFFF     PPPPP         105.77     -0.0080
   19  WFFFF     PPPPP         100.81     -0.0079
   20  WFFFF     PPPPP         101.21     -0.0080
   21  WFFFF     PPPPP          98.17     -0.0082
   22  WFFFF     PPPPF         100.32     -0.0081
   23  WFFFF     PPPPP         102.90     -0.0082
   24  WFFFF     PPPPP         107.21     -0.0081
   25  WFFFF     PPPPP          99.96     -0.0081
   26  WFFFF     PPPPP         101.06     -0.0080
   27  WFFFF     PPPPP         100.35     -0.0080
   28  WFFFF     PPPPP          95.83     -0.0081
   29  WFFFF     PPPPP          97.29     -0.0081
```

## Nota — el modo demo (`--seed`) tropieza con su propio invariant

El `self_check` del demo exige que la TRAMPA no pase ningún check; con la semilla [4] el check 2 (fills al book) devuelve PASS y el modo demo de `audit_demo.py --seed <esa>` terminaría con assert (exit != 0). El Monte Carlo no llama a `self_check` — lo mide. Los checks 1/3/4/5 siguen disparando en ese run: la trampa nunca se aprobó de punta a punta.

## Lectura

- **FP de punta a punta = 0/30**: ninguna semilla aprobó la trampa completa; en las 30, al menos checks 1 (WARN) + 3 + 4 + 5 dispararon.

- **El único miss detector-level sobre la trampa es check 2, y no es ruido estadístico: es el sesgo plantado superando el fee.** En seed [4] el edge look-ahead de la trampa dio +0.0084/trade (vs los ~+0.0049 típicos): mayor que el round trip (2 x 40 bps = 0.008), así que el mean con spread quedó en +0.0004/trade > 0 y "el edge sobrevive el spread" es, por definición del check, cierto (p = 0.85, H1: mean <= 0 no se rechaza porque el mean es positivo). El detector no falló su test; la trampa fue tan lucrativa sobre ese pase aleatorio que soportó el costo. La trampa sigue cazada por checks 1 (WARN), 3, 4 y 5.

- **FN = 1/30 a nivel run — y es calibración, no sesgo del audit**: en seed [22] el check 5 (supervivencia, binomial exacto a alpha=0.01) disparó FAIL sobre el control limpio (asimetría de colas por azar del pase aleatorio). 5 checks x 30 runs = 150 tests bajo la hipótesis nula para el limpio: con alpha=1% se espera ~1.5 falso disparo; se observó 1. Los checks están calibrados, no inflados.

