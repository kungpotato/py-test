import tkinter as tk


def add_task():
    task = task_entry.get()
    if task != "":
        listbox.insert(tk.END, task)
        task_entry.delete(0, tk.END)
    else:
        tk.messagebox.showwarning("Warning", "Please enter a task!")


def delete_task():
    try:
        selected_task_index = listbox.curselection()[0]
        listbox.delete(selected_task_index)
    except:
        tk.messagebox.showwarning("Warning", "Please select a task to delete!")


# Create main window
root = tk.Tk()
root.title("To-Do List")

# Create UI elements
frame = tk.Frame(root)
frame.pack(pady=10)

listbox = tk.Listbox(
    frame,
    width=50,
    height=10,
    bd=0,
    font=("Courier New", 12),
    selectbackground="#a6a6a6"
)
listbox.pack(side=tk.LEFT, fill=tk.BOTH)

scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)

listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=listbox.yview)

task_entry = tk.Entry(
    root,
    font=("Courier New", 12)
)
task_entry.pack(pady=10)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

add_task_button = tk.Button(
    button_frame,
    text="Add Task",
    command=add_task,
    font=("Courier New", 12)
)
add_task_button.pack(side=tk.LEFT)

delete_task_button = tk.Button(
    button_frame,
    text="Delete Task",
    command=delete_task,
    font=("Courier New", 12)
)
delete_task_button.pack(side=tk.LEFT)

# Start the main event loop
root.mainloop()
