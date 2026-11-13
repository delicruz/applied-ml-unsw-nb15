from reporting.report_assets import generate_report_figures


if __name__ == "__main__":
    outputs = generate_report_figures()
    for name, path in outputs.items():
        print(f"{name}: {path}")
