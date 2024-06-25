import numpy as np
import pandas as pd

from omegaconf import DictConfig

from superlinked.evaluation.vector_sampler import VectorSampler
from superlinked.framework.common.embedding.number_embedding import Mode
from superlinked.framework.common.schema.schema import schema
from superlinked.framework.common.schema.schema_object import (
    String,
    Integer,
)
from superlinked.framework.common.schema.id_schema_object import IdField
from superlinked.framework.common.parser.dataframe_parser import DataFrameParser
from superlinked.framework.dsl.executor.in_memory.in_memory_executor import (
    InMemoryExecutor,
    InMemoryApp,
)
from superlinked.framework.dsl.index.index import Index
from superlinked.framework.dsl.source.in_memory_source import InMemorySource
from superlinked.framework.dsl.space.number_space import NumberSpace
from superlinked.framework.dsl.space.text_similarity_space import TextSimilaritySpace
from superlinked.framework.dsl.space.categorical_similarity_space import (
    CategoricalSimilaritySpace,
)

from util.common import camel_to_snake, separate_numeric_and_string_columns


def create_superlinked_embeddings(dataset: pd.DataFrame, config: DictConfig):
    class Employee_skeleteon:
        id: IdField

    numeric_columns, string_columns = separate_numeric_and_string_columns(
        dataset.drop(config.data.target_colname, axis=1)
    )
    for col in numeric_columns:
        Employee_skeleteon.__annotations__[camel_to_snake(col)] = Integer
    for col in string_columns:
        Employee_skeleteon.__annotations__[camel_to_snake(col)] = String

    Employee = schema(Employee_skeleteon)
    employee = Employee()

    spaces_all = [
        CategoricalSimilaritySpace(
            employee.over_time,
            categories=["Yes", "No"],
            uncategorized_as_category=False,
        )
        if "OverTime" in dataset.columns
        else None,
        CategoricalSimilaritySpace(
            employee.marital_status,
            categories=["Married", "Single", "Divorced"],
            uncategorized_as_category=False,
        )
        if "MaritalStatus" in dataset.columns
        else None,
        TextSimilaritySpace(employee.education_field, config.embedding_model.model_name)
        if "EducationField" in dataset.columns
        else None,
        NumberSpace(
            employee.years_in_current_role, min_value=0, max_value=7, mode=Mode.SIMILAR
        )
        if "YearsInCurrentRole" in dataset.columns
        else None,
        NumberSpace(
            employee.years_at_company, min_value=0, max_value=18, mode=Mode.SIMILAR
        )
        if "YearsAtCompany" in dataset.columns
        else None,
        NumberSpace(
            employee.total_working_years, min_value=0, max_value=20, mode=Mode.SIMILAR
        )
        if "TotalWorkingYears" in dataset.columns
        else None,
        NumberSpace(
            employee.job_involvement, min_value=0, max_value=4, mode=Mode.SIMILAR
        )
        if "JobInvolvement" in dataset.columns
        else None,
        NumberSpace(
            employee.stock_option_level, min_value=0, max_value=3, mode=Mode.SIMILAR
        )
        if "StockOptionLevel" in dataset.columns
        else None,
        NumberSpace(
            employee.years_with_curr_manager,
            min_value=0,
            max_value=6,
            mode=Mode.SIMILAR,
        )
        if "YearsWithCurrManager" in dataset.columns
        else None,
        NumberSpace(
            employee.environment_satisfaction,
            min_value=0,
            max_value=4,
            mode=Mode.SIMILAR,
        )
        if "EnvironmentSatisfaction" in dataset.columns
        else None,
        NumberSpace(
            employee.job_satisfaction, min_value=0, max_value=4, mode=Mode.SIMILAR
        )
        if "JobSatisfaction" in dataset.columns
        else None,
    ]
    spaces = [f for f in spaces_all if f is not None]

    skip_columns: list[str] = ["naive_input", config.data.target_colname]

    field_to_column_mapping = {
        getattr(employee, camel_to_snake(col)): col
        for col in dataset.columns
        if col not in skip_columns
    }
    employee_index = Index(spaces=spaces)
    employee_df_parser = DataFrameParser(
        schema=employee, mapping=field_to_column_mapping
    )
    source: InMemorySource = InMemorySource(employee, parser=employee_df_parser)
    executor: InMemoryExecutor = InMemoryExecutor(
        sources=[source], indices=[employee_index]
    )
    app: InMemoryApp = executor.run()

    df_to_put = dataset.reset_index().rename(columns={"index": "id"})
    source.put([df_to_put])

    # collect all vectors from the app
    vs = VectorSampler(app=app)
    vector_collection = vs.get_all_vectors(employee_index, employee)
    paired = list(zip(map(int, vector_collection.id_list), vector_collection.vectors))
    paired.sort()
    sorted_id_list, sorted_vectors = zip(*paired)
    return np.stack(sorted_vectors)
