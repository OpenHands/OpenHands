import ast
import logging
import re
import traceback
from typing import Any

import numpy as np
from sympy import Rational

from tasks.base import Task

LOGGER = logging.getLogger('MINT')


class ReasoningTask(Task):
    task_name = 'reasoning'

    def __init__(self, id: str, prompt: str, reference: str, **kwargs):
        super().__init__(**kwargs)
        self._id = id
        self._prompt = prompt.strip()
        self._reference = str(reference).strip().lower()

    def extract_answer(self, solution: str) -> str | None:
        """Extract the answer from the given solution."""
        return solution.lower().strip()

    def compare_w_digits(self, reference: str, answer: str) -> bool:
        """Compare the reference and answer with digits."""
        # if reference can and answer can both be converted to floats by float()
        try:
            float(reference)
            float(answer)
            return abs(float(reference) - float(answer)) <= 0.05 * abs(float(reference))
        except ValueError:
            return reference in answer
        except Exception:
            raise ValueError(f'Cannot compare {reference} and {answer}')

    def success(self, solution: str) -> bool:
        answer = self.extract_answer(solution)
        return self.compare_w_digits(self._reference, answer)


class MultipleChoiceTask(Task):
    """Subclass of Task for multiple choice tasks."""

    task_name = 'reasoning'

    def __init__(self, id, prompt: str, reference: str, **kwargs):
        super().__init__(**kwargs)
        self._id = id
        self.hide_options = kwargs.get('hide_options', False)
        if self.hide_options:
            self._prompt = prompt.split('Options:')[0].strip()
        else:
            self._prompt = prompt
        self._reference = reference.strip().lower()
        self._options = self.extract_options(prompt)
        # if all options can be converted to float, strictly perform hide options
        try:
            for option in self._options.values():
                float(option)
            self.hide_options = True
        except ValueError:
            pass
        self.metadata.update({'options': self._options})

    def extract_answer(self, solution: str) -> str | None:
        # Extract the selected option from the solution
        solution = solution.lower().strip()
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if f'{letter})' in solution or f'{letter} )' in solution:
                print('SOLUTION', letter)
                return letter
            else:
                print('SOLUTION', solution)
                return solution

    def compare_w_digits(self, reference: str, answer: str) -> bool:
        if reference.isdigit() and answer.isdigit():
            return abs(float(reference) - float(answer)) <= 0.05 * float(reference)
        else:
            return reference in answer

    def success(self, solution: str) -> bool:
        answer = self.extract_answer(solution)
        if self.compare_w_digits(self._reference, answer):
            return True
        else:
            correct_option = self._options[self._reference]
            wrong_option_list = list(self._options.values())
            print('OPTIONS', correct_option, wrong_option_list)
            print('ANSWER', answer)
            for i in wrong_option_list:
                if i in correct_option:
                    wrong_option_list.remove(i)
            for i in wrong_option_list:
                if self.compare_w_digits(i, answer) or (i in answer):
                    return False
            if self.compare_w_digits(correct_option, answer) or (
                correct_option in answer
            ):
                return True
            else:
                return False

    def extract_options(self, prompt: str) -> dict:
        # Find the possible option separators (comma, semicolon, or parentheses)
        prompt = prompt.split('Options: ')[-1]
        # Extract the options using the delimiter
        options_match = prompt.split(' , ')
        options = {}
        for i in range(len(options_match)):
            option = options_match[i].strip("[]' ")
            option = option.split(')')
            letter = option[0].lower().strip()
            content = (
                option[1]
                .lower()
                .strip('.')
                .replace('. Which option is correct?', '')
                .replace('. Which one is correct?', '')
                .strip()
            )
            options.update({letter: content})
        return options


# ==== TheoremQA ====


def compare_two_numbers(p, gt):
    if isinstance(p, (int, float)):
        pass
    elif isinstance(p, (bool, complex, dict, list, str, tuple)):
        return False
    else:
        raise ValueError(p)

    if isinstance(gt, float):
        return within_eps(pred=p, gt=gt)
    else:
        return round(p) == gt


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    elif len(pred) != len(gt):
        return False
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([compare_two_numbers(p, g) for p, g in zip(pred, gt)])


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def parse_number_list(s: str):
    # Check if the string is a valid list by trying to parse it
    parsed_list = ast.literal_eval(s)
    return parsed_list


def is_number(string):
    pattern = r'^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$'
    match = re.match(pattern, string)
    return bool(match)


def is_scientific_number(string):
    pattern = r'^[-+]?\d+(\.\d+)?e[-]?\d+$'
    match = re.match(pattern, string)
    return bool(match)


def contain_num_and_str(string):
    pattern_str = r'[a-zA-Z]'
    pattern_num = r'[0-9]'
    return bool(re.search(pattern_str, string) and re.search(pattern_num, string))


class TheoremqaTask(Task):
    task_name = 'reasoning'

    def __init__(self, id: str, prompt: str, reference: str, **kwargs):
        super().__init__(**kwargs)
        self._id = id
        self._prompt = (
            'Answer the following question with a number, a list of numbers or True or False. '
            + prompt.strip()
        )
        self._reference = reference
        self._answer_type = kwargs.get('answer_type')

    def extract_answer(self, solution: str) -> Any:
        """Extract the answer from the given solution."""
        prediction = solution
        # Following the preprocessing steps from TheoremQA
        # https://github.com/wenhuchen/TheoremQA/blob/123e36beaaa97c01f28a582f13c4f77a6822c199/predict_accuracy.py#L170

        # Preprocessing the string [Stage 1]
        if not isinstance(prediction, str):
            prediction = str(prediction) if prediction is not None else '0'

        # Replace special tokens
        if '=' in prediction:
            prediction = prediction.split('=')[-1].strip()
        if '≈' in prediction:
            prediction = prediction.split('≈')[-1].strip()
        if '`' in prediction:
            prediction = prediction.replace('`', '')
        if '$' in prediction:
            prediction = prediction.replace('$', '')
        if '°' in prediction:
            prediction = prediction.replace('°', '')

        # Detect the boolean keyword in the generation
        if prediction in ('true', 'yes', 'false', 'no'):
            if prediction in ('true', 'yes'):
                prediction = 'True'
            else:
                prediction = 'False'
        if 'True' in prediction or 'False' in prediction:
            prediction = 'True' if 'True' in prediction else 'False'

        # Detect the approximation keyword
        if 'approximately' in prediction:
            prediction = prediction.replace('approximately', '').strip()
        if ' or ' in prediction:
            prediction = prediction.split(' or ')[0]

        # Drop the units before and after the number
        if re.match(r'[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$', prediction):
            prediction = re.search(
                r'([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$', prediction
            ).group(1)
        if re.match(r'[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$', prediction):
            prediction = re.search(
                r'[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$', prediction
            ).group(1)
        if re.match(r'[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$', prediction):
            prediction = re.search(
                r'([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$', prediction
            ).group(1)
        if re.match(r'[^-+\d]{1,2}(?:[\d,]*\.*\d+)$', prediction):
            prediction = re.search(
                r'[^-+\d]{1,2}((?:[\d,]*\.*\d+))$', prediction
            ).group(1)

        # Preprocessing the number [Stage 1]
        if '10^' in prediction:
            prediction = re.sub(r'10\^(-?\d+)', r'math.pow(10, \1)', prediction)
        if ' x ' in prediction:
            prediction = prediction.replace(' x ', '*')
        if ' × ' in prediction:
            prediction = prediction.replace(' × ', '*')
        if is_number(prediction):
            prediction = prediction.replace(',', '')

        # Preprocessing the option [Stage 3]
        if (
            'a)' in prediction
            or 'a )' in prediction
            or prediction.lower().strip() == 'a'
        ):
            prediction = '(a)'
        if (
            'b)' in prediction
            or 'b )' in prediction
            or prediction.lower().strip() == 'b'
        ):
            prediction = '(b)'
        if (
            'c)' in prediction
            or 'c )' in prediction
            or prediction.lower().strip() == 'c'
        ):
            prediction = '(c)'
        if (
            'd)' in prediction
            or 'd )' in prediction
            or prediction.lower().strip() == 'd'
        ):
            prediction = '(d)'

        if (
            '(a)' in prediction
            or '(b)' in prediction
            or '(c)' in prediction
            or '(d)' in prediction
        ):
            prediction = '"' + re.search(r'\([a-d]\)', prediction).group(0) + '"'

        # If the prediction is empty, use dummy '0'
        if not prediction:
            prediction = '0'

        # Converting the string answer to a number/list/bool/option
        try:
            prediction = ast.literal_eval(prediction)
        except Exception:
            LOGGER.warning(
                f'[TASK] Failed to convert the answer: {prediction}\n{traceback.format_exc()}'
            )
            return None  # failed to convert the answer

        # Performing common type conversion
        if isinstance(prediction, (set, tuple)):
            prediction = list(prediction)
            if isinstance(prediction[0], complex):
                prediction = [tmp.real for tmp in prediction]
            elif isinstance(prediction[0], Rational):
                prediction = [float(tmp) for tmp in prediction]
        elif isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        else:
            if isinstance(prediction, complex):
                prediction = prediction.real
            elif isinstance(prediction, Rational):
                prediction = float(prediction)

        return prediction

    def success(self, solution: str) -> bool:
        """This checks whether the given solution can complete the current task."""
        # Follow the implementation from TheoremQA
        # https://github.com/wenhuchen/TheoremQA/blob/123e36beaaa97c01f28a582f13c4f77a6822c199/predict_accuracy.py#L301C9-L317C1
        prediction = self.extract_answer(solution)
        LOGGER.info(f'TheoremQA Parsed Prediction: {prediction}')
        answer_type = self._answer_type
        gt = self.extract_answer(self.reference)

        if isinstance(prediction, (str, int, float, list)):
            # Comparing prediction against the reference
            if answer_type in ['bool', 'option', 'Option']:
                cur_correct = int(prediction == f'({gt})') or int(prediction == gt)
            elif answer_type == 'integer':
                cur_correct = int(compare_two_numbers(prediction, gt))
            elif answer_type == 'float':
                cur_correct = int(compare_two_numbers(prediction, gt))
            elif answer_type in ['list of integer', 'list of float']:
                cur_correct = int(compare_two_list(prediction, gt))
        else:
            cur_correct = 0
        return bool(cur_correct)
