import importlib

for m in ["torch", "xlearn", "lightfm", "fastFM", "tensorflow",
          "sklearn", "catboost", "scipy", "numpy", "pandas"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<12} OK   {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<12} --   {type(e).__name__}")
