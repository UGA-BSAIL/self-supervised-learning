# coding: utf-8

"""
    CVAT REST API

    REST API for Computer Vision Annotation Tool (CVAT)  # noqa: E501

    OpenAPI spec version: v1
    Contact: nikita.manovich@intel.com
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

from __future__ import absolute_import

import unittest

import swagger_client
from swagger_client.api.users_api import UsersApi  # noqa: E501
from swagger_client.rest import ApiException


class TestUsersApi(unittest.TestCase):
    """UsersApi unit test stubs"""

    def setUp(self):
        self.api = UsersApi()  # noqa: E501

    def tearDown(self):
        pass

    def test_users_delete(self):
        """Test case for users_delete

        Method deletes a specific user from the server  # noqa: E501
        """
        pass

    def test_users_list(self):
        """Test case for users_list

        Method provides a paginated list of users registered on the server  # noqa: E501
        """
        pass

    def test_users_partial_update(self):
        """Test case for users_partial_update

        Method updates chosen fields of a user  # noqa: E501
        """
        pass

    def test_users_read(self):
        """Test case for users_read

        Method provides information of a specific user  # noqa: E501
        """
        pass

    def test_users_self(self):
        """Test case for users_self

        Method returns an instance of a user who is currently authorized  # noqa: E501
        """
        pass


if __name__ == "__main__":
    unittest.main()
