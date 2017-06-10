R"(

import opentuner
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.measurement.interface import DefaultMeasurementInterface
from opentuner.api import TuningRunManager
from opentuner.resultsdb.models import Result
from opentuner.search.manipulator import IntegerParameter
import argparse

def get_next_desired_result():
    global desired_result
    desired_result = api.get_next_desired_result()
    while desired_result is None:
        desired_result = api.get_next_desired_result()
    return desired_result.configuration.data

def report_result(runtime):
    api.report_result(desired_result, Result(time=runtime))

def finish():
    api.finish()

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
args = parser.parse_args()

manipulator = ConfigurationManipulator()
:::parameters:::
interface = DefaultMeasurementInterface(args=args,
                                        manipulator=manipulator,
                                        project_name='atf_library',
                                        program_name='atf_library',
                                        program_version='0.1')


api = TuningRunManager(interface, args)

)"
