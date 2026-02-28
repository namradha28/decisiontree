import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

if __name__ == "__main__":
    print("Launching minimal demo...")
    try:
        demo.launch()
    except Exception as e:
        print(f"Launch failed: {e}")
