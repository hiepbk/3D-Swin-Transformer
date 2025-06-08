import time

def track_progress(iterable, description="Progress", bar_length=20, show_eta=True):
    """
    Track progress of any iterable with progress display
    
    Args:
        iterable: Any iterable (dataloader, list, etc.)
        description (str): Description text to show
        bar_length (int): Length of progress bar
        show_eta (bool): Whether to show estimated time of arrival
        
    Yields:
        tuple: (index, item) for each item in iterable
        
    Example:
        for i, data in track_progress(dataloader, "Training"):
            # Your code here
            pass
    """
    total = len(iterable) if hasattr(iterable, '__len__') else None
    start_time = time.time()
    
    try:
        for i, item in enumerate(iterable):
            # Show progress before yielding
            if total:
                progress = (i + 1) / total
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                percentage = progress * 100
                
                progress_str = f"{description}: {i+1}/{total} |{bar}| {percentage:.1f}%"
                
                if show_eta and i > 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / (i + 1)) * (total - (i + 1))
                    progress_str += f" ETA: {eta:.1f}s"
                    
                print(progress_str, end='\r')
            else:
                # Unknown total length
                spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                spin_char = spinner[i % 10]
                elapsed = time.time() - start_time
                print(f"{spin_char} {description}: {i+1} items | {elapsed:.1f}s", end='\r')
            
            yield i, item
            
    finally:
        # Clear progress line when done
        print("\r" + " " * 80 + "\r", end="")
        elapsed = time.time() - start_time
        if total:
            print(f"{description}: {total}/{total} completed in {elapsed:.1f}s")
        else:
            print(f"{description}: {i+1} items completed in {elapsed:.1f}s")
