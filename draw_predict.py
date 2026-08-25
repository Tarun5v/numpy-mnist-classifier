"""
Simple tkinter GUI for drawing digits and predicting with the trained model.
Draw a digit (0-9) on the canvas, click Predict, and see what the model thinks.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from neural_network import NeuralNetwork


class DigitRecognizer:
    """GUI application for recognizing handwritten digits."""
    
    def __init__(self, model_path='models/mnist_model.npy'):
        """Initialize the GUI and load the trained model."""
        self.root = tk.Tk()
        self.root.title("MNIST Digit Recognizer")
        self.root.resizable(False, False)
        
        # Load the trained model
        print("Loading trained model...")
        self.model = NeuralNetwork.load_weights(model_path)
        
        # Canvas settings
        self.canvas_size = 280  # 10x the MNIST size (28px)
        self.brush_size = 15
        
        # Create drawing image (28x28 for MNIST)
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the GUI layout."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Draw a Digit (0-9)", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Canvas for drawing
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=1, column=0, padx=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size,
                               bg='black', cursor='cross')
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_position)
        
        # Right side panel
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1)
        
        # Prediction display
        pred_label = ttk.Label(right_frame, text="Prediction:", 
                              font=('Helvetica', 12, 'bold'))
        pred_label.grid(row=0, column=0, pady=(0, 5))
        
        self.prediction_var = tk.StringVar(value="?")
        self.prediction_label = ttk.Label(right_frame, textvariable=self.prediction_var,
                                         font=('Helvetica', 48, 'bold'),
                                         foreground='blue')
        self.prediction_label.grid(row=1, column=0, pady=(0, 10))
        
        # Confidence display
        conf_label = ttk.Label(right_frame, text="Confidence:", 
                              font=('Helvetica', 10))
        conf_label.grid(row=2, column=0)
        
        self.confidence_var = tk.StringVar(value="0%")
        self.confidence_display = ttk.Label(right_frame, textvariable=self.confidence_var,
                                           font=('Helvetica', 14))
        self.confidence_display.grid(row=3, column=0, pady=(0, 20))
        
        # Buttons
        predict_btn = ttk.Button(right_frame, text="Predict", 
                                command=self.predict_digit)
        predict_btn.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        clear_btn = ttk.Button(right_frame, text="Clear", 
                              command=self.clear_canvas)
        clear_btn.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Probability bars
        prob_label = ttk.Label(right_frame, text="Probabilities:", 
                              font=('Helvetica', 10, 'bold'))
        prob_label.grid(row=6, column=0, pady=(10, 5))
        
        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            frame = ttk.Frame(right_frame)
            frame.grid(row=7+i, column=0, sticky=(tk.W, tk.E), pady=1)
            
            digit_label = ttk.Label(frame, text=f"{i}:", width=2)
            digit_label.pack(side=tk.LEFT)
            
            bar = ttk.Progressbar(frame, length=100, mode='determinate')
            bar.pack(side=tk.LEFT, padx=(5, 5))
            
            val_label = ttk.Label(frame, text="0%", width=5)
            val_label.pack(side=tk.LEFT)
            
            self.prob_bars.append(bar)
            self.prob_labels.append(val_label)
        
        # Instructions
        inst_frame = ttk.Frame(main_frame)
        inst_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        inst_text = ttk.Label(inst_frame, 
                             text="Draw with mouse • Click Predict to classify • Click Clear to reset",
                             font=('Helvetica', 9),
                             foreground='gray')
        inst_text.pack()
    
    def paint(self, event):
        """Draw on the canvas when mouse is dragged."""
        x, y = event.x, event.y
        r = self.brush_size // 2
        self.canvas.create_ellipse(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    def reset_position(self, event):
        """Reset drawing position after mouse release."""
        pass
    
    def clear_canvas(self):
        """Clear the canvas and reset the drawing."""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_var.set("?")
        self.confidence_var.set("0%")
        for i in range(10):
            self.prob_bars[i]['value'] = 0
            self.prob_labels[i].config(text="0%")
    
    def predict_digit(self):
        """Predict the drawn digit using the trained model."""
        # Resize to 28x28 and normalize
        img_28 = self.image.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img_28).astype(np.float64) / 255.0
        
        # Reshape to (1, 784) for the model
        img_flat = img_array.reshape(1, -1)
        
        # Check if canvas is empty (all black)
        if np.sum(img_array) < 10:
            self.prediction_var.set("?")
            self.confidence_var.set("Draw something!")
            return
        
        # Get prediction
        probabilities = self.model.predict_proba(img_flat)[0]
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[predicted_digit]
        
        # Update display
        self.prediction_var.set(str(predicted_digit))
        self.confidence_var.set(f"{confidence:.1%}")
        
        # Update probability bars
        for i in range(10):
            self.prob_bars[i]['value'] = probabilities[i] * 100
            self.prob_labels[i].config(text=f"{probabilities[i]:.1%}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


def main():
    """Launch the digit recognizer GUI."""
    try:
        app = DigitRecognizer()
        app.run()
    except FileNotFoundError:
        print("Error: No trained model found at models/mnist_model.npy")
        print("Please run main.py first to train the model.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have PIL/Pillow installed: pip install Pillow")


if __name__ == '__main__':
    main()
