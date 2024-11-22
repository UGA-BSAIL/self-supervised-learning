# BodyQueryArtifactsQueryPost


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**queries** | [**List[ImageQuery]**](ImageQuery.md) |  | [optional] [default to [{}]]
**orderings** | [**List[Ordering]**](Ordering.md) |  | [optional] [default to []]

## Example

```python
from mallard_client.models.body_query_artifacts_query_post import BodyQueryArtifactsQueryPost

# TODO update the JSON string below
json = "{}"
# create an instance of BodyQueryArtifactsQueryPost from a JSON string
body_query_artifacts_query_post_instance = BodyQueryArtifactsQueryPost.from_json(json)
# print the JSON string representation of the object
print BodyQueryArtifactsQueryPost.to_json()

# convert the object into a dict
body_query_artifacts_query_post_dict = body_query_artifacts_query_post_instance.to_dict()
# create an instance of BodyQueryArtifactsQueryPost from a dict
body_query_artifacts_query_post_form_dict = body_query_artifacts_query_post.from_dict(body_query_artifacts_query_post_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
