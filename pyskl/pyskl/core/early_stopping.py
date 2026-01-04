# Copyright (c) OpenMMLab. All rights reserved.
"""Early stopping hook for MMCV runners.

This module implements early stopping functionality to prevent overfitting
by monitoring validation metrics and stopping training when performance
stops improving.

IMPORTANT: This hook must run AFTER EvalHook in after_train_epoch, because
EvalHook stores metrics in runner.log_buffer.output during after_train_epoch.
Register this hook with priority='LOW' (60) to ensure correct ordering.
"""

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Early stopping hook that monitors validation metrics.
    
    Stops training when the monitored metric has not improved for a
    specified number of epochs (patience).
    
    NOTE: This hook checks metrics in after_train_epoch (not after_val_epoch)
    because MMCV's EvalHook stores metrics in log_buffer.output during
    after_train_epoch. The hook should be registered with priority='LOW'
    to ensure it runs after EvalHook.
    
    Args:
        patience (int): Number of epochs to wait for improvement before
            stopping. Default: 3.
        monitor (str): Name of the metric to monitor. Should match the
            metric name in evaluation results. Default: 'top1_acc'.
        min_delta (float): Minimum change to qualify as an improvement.
            Default: 0.0.
        mode (str): One of {'max', 'min'}. In 'max' mode, training stops
            when the metric stops increasing; in 'min' mode, training
            stops when the metric stops decreasing. Default: 'max'.
        verbose (bool): If True, print messages when early stopping
            conditions are checked. Default: True.
    
    Example:
        >>> # In config file:
        >>> early_stopping = dict(
        ...     patience=3,
        ...     monitor='top1_acc',
        ...     min_delta=0.0)
    """

    def __init__(self,
                 patience=3,
                 monitor='top1_acc',
                 min_delta=0.0,
                 mode='max',
                 verbose=True):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        # State tracking
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.last_checked_epoch = -1
        
        # Adjust min_delta sign based on mode
        if mode == 'min':
            self.min_delta *= -1
            self.monitor_op = lambda a, b: a < b - self.min_delta
        else:  # mode == 'max'
            self.monitor_op = lambda a, b: a > b + self.min_delta

    def before_run(self, runner):
        """Initialize state before training starts."""
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.last_checked_epoch = -1
        if self.verbose:
            runner.logger.info(
                f'EarlyStoppingHook: monitoring "{self.monitor}" with '
                f'patience={self.patience}, mode={self.mode}, priority={self.priority}')

    def after_train_epoch(self, runner):
        """Check if validation metric improved after each training epoch.
        
        This runs AFTER EvalHook.after_train_epoch(), which stores validation
        metrics in runner.log_buffer.output. We check for metrics here because
        that's when they're available.
        
        Args:
            runner: The runner object that manages training.
        """
        # Avoid checking the same epoch twice
        if runner.epoch == self.last_checked_epoch:
            return
        
        # Check if we should stop from previous epoch
        if self.should_stop:
            runner._max_epochs = runner.epoch + 1
            if self.verbose:
                runner.logger.info(
                    f'EarlyStoppingHook: training will stop after epoch {runner.epoch + 1}')
            return
        
        # Get the current metric value from runner.log_buffer
        # EvalHook stores metrics in runner.log_buffer.output during after_train_epoch
        if not hasattr(runner, 'log_buffer') or not hasattr(runner.log_buffer, 'output'):
            return
        
        log_output = runner.log_buffer.output
        
        # Try different possible metric key formats
        current_score = None
        possible_keys = [
            self.monitor,
            f'val_{self.monitor}',
        ]
        
        for key in possible_keys:
            if key in log_output:
                current_score = log_output[key]
                break
        
        if current_score is None:
            # Metrics not available - validation might not have run this epoch
            # Only warn if we have keys in log_output (i.e., something was logged)
            if self.verbose and len(log_output) > 0:
                available_keys = list(log_output.keys())
                # Only warn if this looks like it should have validation metrics
                if 'eval_iter_num' in log_output or any('acc' in k for k in available_keys):
                    runner.logger.warning(
                        f'EarlyStoppingHook: metric "{self.monitor}" not found. '
                        f'Available metrics: {available_keys}')
            return
        
        self.last_checked_epoch = runner.epoch
        
        # Check if this is an improvement
        if self.best_score is None:
            self.best_score = current_score
            if self.verbose:
                runner.logger.info(
                    f'EarlyStoppingHook: initial {self.monitor}={current_score:.4f}')
        elif self.monitor_op(current_score, self.best_score):
            # Improvement found
            if self.verbose:
                runner.logger.info(
                    f'EarlyStoppingHook: {self.monitor} improved from '
                    f'{self.best_score:.4f} to {current_score:.4f}')
            self.best_score = current_score
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                runner.logger.info(
                    f'EarlyStoppingHook: no improvement in {self.monitor} '
                    f'for {self.counter}/{self.patience} epochs '
                    f'(best={self.best_score:.4f}, current={current_score:.4f})')
            
            if self.counter >= self.patience:
                self.should_stop = True
                runner.logger.info(
                    f'EarlyStoppingHook: STOPPING - no improvement in {self.monitor} '
                    f'for {self.patience} epochs (best={self.best_score:.4f})')
                # Stop immediately by setting max_epochs
                runner._max_epochs = runner.epoch + 1
