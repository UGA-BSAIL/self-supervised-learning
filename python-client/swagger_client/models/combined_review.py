# coding: utf-8

"""
    CVAT REST API

    REST API for Computer Vision Annotation Tool (CVAT)  # noqa: E501

    OpenAPI spec version: v1
    Contact: nikita.manovich@intel.com
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six


class CombinedReview(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        "id": "int",
        "assignee": "BasicUser",
        "assignee_id": "int",
        "reviewer": "BasicUser",
        "reviewer_id": "int",
        "issue_set": "list[CombinedIssue]",
        "estimated_quality": "float",
        "status": "str",
        "job": "int",
    }

    attribute_map = {
        "id": "id",
        "assignee": "assignee",
        "assignee_id": "assignee_id",
        "reviewer": "reviewer",
        "reviewer_id": "reviewer_id",
        "issue_set": "issue_set",
        "estimated_quality": "estimated_quality",
        "status": "status",
        "job": "job",
    }

    def __init__(
        self,
        id=None,
        assignee=None,
        assignee_id=None,
        reviewer=None,
        reviewer_id=None,
        issue_set=None,
        estimated_quality=None,
        status=None,
        job=None,
    ):  # noqa: E501
        """CombinedReview - a model defined in Swagger"""  # noqa: E501
        self._id = None
        self._assignee = None
        self._assignee_id = None
        self._reviewer = None
        self._reviewer_id = None
        self._issue_set = None
        self._estimated_quality = None
        self._status = None
        self._job = None
        self.discriminator = None
        if id is not None:
            self.id = id
        if assignee is not None:
            self.assignee = assignee
        if assignee_id is not None:
            self.assignee_id = assignee_id
        if reviewer is not None:
            self.reviewer = reviewer
        if reviewer_id is not None:
            self.reviewer_id = reviewer_id
        self.issue_set = issue_set
        self.estimated_quality = estimated_quality
        self.status = status
        self.job = job

    @property
    def id(self):
        """Gets the id of this CombinedReview.  # noqa: E501


        :return: The id of this CombinedReview.  # noqa: E501
        :rtype: int
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this CombinedReview.


        :param id: The id of this CombinedReview.  # noqa: E501
        :type: int
        """

        self._id = id

    @property
    def assignee(self):
        """Gets the assignee of this CombinedReview.  # noqa: E501


        :return: The assignee of this CombinedReview.  # noqa: E501
        :rtype: BasicUser
        """
        return self._assignee

    @assignee.setter
    def assignee(self, assignee):
        """Sets the assignee of this CombinedReview.


        :param assignee: The assignee of this CombinedReview.  # noqa: E501
        :type: BasicUser
        """

        self._assignee = assignee

    @property
    def assignee_id(self):
        """Gets the assignee_id of this CombinedReview.  # noqa: E501


        :return: The assignee_id of this CombinedReview.  # noqa: E501
        :rtype: int
        """
        return self._assignee_id

    @assignee_id.setter
    def assignee_id(self, assignee_id):
        """Sets the assignee_id of this CombinedReview.


        :param assignee_id: The assignee_id of this CombinedReview.  # noqa: E501
        :type: int
        """

        self._assignee_id = assignee_id

    @property
    def reviewer(self):
        """Gets the reviewer of this CombinedReview.  # noqa: E501


        :return: The reviewer of this CombinedReview.  # noqa: E501
        :rtype: BasicUser
        """
        return self._reviewer

    @reviewer.setter
    def reviewer(self, reviewer):
        """Sets the reviewer of this CombinedReview.


        :param reviewer: The reviewer of this CombinedReview.  # noqa: E501
        :type: BasicUser
        """

        self._reviewer = reviewer

    @property
    def reviewer_id(self):
        """Gets the reviewer_id of this CombinedReview.  # noqa: E501


        :return: The reviewer_id of this CombinedReview.  # noqa: E501
        :rtype: int
        """
        return self._reviewer_id

    @reviewer_id.setter
    def reviewer_id(self, reviewer_id):
        """Sets the reviewer_id of this CombinedReview.


        :param reviewer_id: The reviewer_id of this CombinedReview.  # noqa: E501
        :type: int
        """

        self._reviewer_id = reviewer_id

    @property
    def issue_set(self):
        """Gets the issue_set of this CombinedReview.  # noqa: E501


        :return: The issue_set of this CombinedReview.  # noqa: E501
        :rtype: list[CombinedIssue]
        """
        return self._issue_set

    @issue_set.setter
    def issue_set(self, issue_set):
        """Sets the issue_set of this CombinedReview.


        :param issue_set: The issue_set of this CombinedReview.  # noqa: E501
        :type: list[CombinedIssue]
        """
        if issue_set is None:
            raise ValueError(
                "Invalid value for `issue_set`, must not be `None`"
            )  # noqa: E501

        self._issue_set = issue_set

    @property
    def estimated_quality(self):
        """Gets the estimated_quality of this CombinedReview.  # noqa: E501


        :return: The estimated_quality of this CombinedReview.  # noqa: E501
        :rtype: float
        """
        return self._estimated_quality

    @estimated_quality.setter
    def estimated_quality(self, estimated_quality):
        """Sets the estimated_quality of this CombinedReview.


        :param estimated_quality: The estimated_quality of this CombinedReview.  # noqa: E501
        :type: float
        """
        if estimated_quality is None:
            raise ValueError(
                "Invalid value for `estimated_quality`, must not be `None`"
            )  # noqa: E501

        self._estimated_quality = estimated_quality

    @property
    def status(self):
        """Gets the status of this CombinedReview.  # noqa: E501


        :return: The status of this CombinedReview.  # noqa: E501
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status):
        """Sets the status of this CombinedReview.


        :param status: The status of this CombinedReview.  # noqa: E501
        :type: str
        """
        if status is None:
            raise ValueError(
                "Invalid value for `status`, must not be `None`"
            )  # noqa: E501
        allowed_values = [
            "accepted",
            "rejected",
            "review_further",
        ]  # noqa: E501
        if status not in allowed_values:
            raise ValueError(
                "Invalid value for `status` ({0}), must be one of {1}".format(  # noqa: E501
                    status, allowed_values
                )
            )

        self._status = status

    @property
    def job(self):
        """Gets the job of this CombinedReview.  # noqa: E501


        :return: The job of this CombinedReview.  # noqa: E501
        :rtype: int
        """
        return self._job

    @job.setter
    def job(self, job):
        """Sets the job of this CombinedReview.


        :param job: The job of this CombinedReview.  # noqa: E501
        :type: int
        """
        if job is None:
            raise ValueError(
                "Invalid value for `job`, must not be `None`"
            )  # noqa: E501

        self._job = job

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(
                    map(
                        lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                        value,
                    )
                )
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value
        if issubclass(CombinedReview, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, CombinedReview):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
